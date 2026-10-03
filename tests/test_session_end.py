"""播放会话结束接口测试。

POST /sessions/{session_id}/end：以一次完整操作取回最终快照并释放会话。
覆盖：成功响应沿用创建会话的全部字段、快照取自轮到结束生效时的会话
（不读取尚未接纳的场景修订）、结束后该 id 一律 404（含已取得会话引用
但尚未生效的在途请求）、并发结束最多一次成功、源场景/其他会话/此前
返回的快照不受影响，以及结束与等待中的推进、事件推进、接纳和重复结束
的可控交错。
"""
import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.playback import PlaybackManager, SessionEnded  # noqa: E402
from app.storage import SceneRecord  # noqa: E402


def scene_payload(track_count=2):
    """s1[0,1000) s2[800,3000)锁2 s3[1000,2500) —— 分轨 1/2/1，换轨 0。"""
    return {
        "track_count": track_count,
        "expected_revision": 0,
        "subtitles": [
            {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
            {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 2500},
        ],
    }


def revised_subtitles_v2():
    """版本 2：s3 延长到 4000，新增 s4[3200,4000)（均在游标 900 之后开始）。"""
    return [
        {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
        {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
         "locked_track": 2},
        {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 4000},
        {"id": "s4", "speaker": "Carol", "start_ms": 3200, "end_ms": 4000},
    ]


def make_session(client, scene_id="scene-end"):
    """创建场景（revision=1）并以其创建会话，返回会话状态。"""
    resp = client.put(f"/scenes/{scene_id}", json=scene_payload())
    assert resp.status_code == 200
    resp = client.post("/sessions", json={"scene_id": scene_id})
    assert resp.status_code == 200
    return resp.json()


def revise(client, scene_id, expected_revision, subtitles, track_count=2):
    body = {
        "track_count": track_count,
        "expected_revision": expected_revision,
        "subtitles": subtitles,
    }
    resp = client.put(f"/scenes/{scene_id}", json=body)
    assert resp.status_code == 200
    assert resp.json()["feasible"] is True
    return resp.json()


def post_async(path, responses, key, json_body=None):
    """在独立线程中发起 POST（独立 TestClient），响应写入 responses[key]。"""
    def target():
        local = TestClient(app)
        if json_body is None:
            responses[key] = local.post(path)
        else:
            responses[key] = local.post(path, json=json_body)

    t = threading.Thread(target=target)
    t.start()
    return t


def patch_first_manager_call(monkeypatch, name):
    """故障注入：指定 PlaybackManager 方法的首次调用在生效前受控暂停。

    暂停点位于路由已取得会话引用之后、操作进入会话锁之前，即构造一个
    “已取得会话引用但尚未生效”的在途请求。返回 (已暂停事件, 放行事件)。
    """
    manager = main_module.get_playback()
    real = getattr(manager, name)
    paused = threading.Event()
    release = threading.Event()
    injected = []

    def wrapped(*args, **kwargs):
        if injected:
            return real(*args, **kwargs)
        injected.append(True)
        paused.set()
        assert release.wait(timeout=10), "测试未能及时放行被暂停的调用"
        return real(*args, **kwargs)

    monkeypatch.setattr(manager, name, wrapped)
    return paused, release


def patch_first_scene_read(monkeypatch):
    """故障注入：让第一次场景读取在生效前受控暂停（放行后才真正读取）。

    接纳的场景读取发生在会话锁内，因此暂停期间会话锁由该接纳请求持有。
    返回 (已暂停事件, 放行事件)。
    """
    store = main_module.get_store()
    real_get_scene = store.get_scene
    paused = threading.Event()
    release = threading.Event()
    injected = []

    def wrapped_get_scene(scene_id):
        if injected:
            return real_get_scene(scene_id)
        injected.append(scene_id)
        paused.set()
        assert release.wait(timeout=10), "测试未能及时放行被暂停的读取"
        return real_get_scene(scene_id)

    monkeypatch.setattr(store, "get_scene", wrapped_get_scene)
    return paused, release


# ---------------------------------------------------------------------------
# 结束：基本语义
# ---------------------------------------------------------------------------


def test_end_returns_creation_response_fields_with_current_state():
    """成功响应沿用创建会话的全部字段，内容取自轮到结束生效时的会话。"""
    client = TestClient(app)
    scene_id = "scene-end"
    created = make_session(client, scene_id)
    sid = created["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    # 场景修订到版本 2 并让会话接纳，使游标与源版本都离开初始值
    revise(client, scene_id, 1, revised_subtitles_v2())
    assert client.post(f"/sessions/{sid}/accept").status_code == 200

    resp = client.post(f"/sessions/{sid}/end")
    assert resp.status_code == 200
    body = resp.json()
    # 字段集合与创建会话的响应完全一致
    assert set(body) == set(created) == {
        "session_id", "scene_id", "source_revision", "cursor_ms",
        "track_count", "changes", "subtitles", "assignments",
    }
    assert body["session_id"] == sid
    assert body["scene_id"] == scene_id
    assert body["source_revision"] == 2  # 轮到结束生效时的会话源版本
    assert body["cursor_ms"] == 900
    assert body["track_count"] == 2
    assert [s["id"] for s in body["subtitles"]] == ["s1", "s2", "s3", "s4"]
    assert body["assignments"] == [
        {"id": "s1", "track": 1},
        {"id": "s2", "track": 2},
        {"id": "s3", "track": 1},
        {"id": "s4", "track": 2},
    ]


def test_end_unknown_session_404():
    client = TestClient(app)
    resp = client.post("/sessions/nope/end")
    assert resp.status_code == 404


def test_end_then_every_operation_on_that_id_404():
    """结束后该 id 的推进/事件推进/接纳/再次结束一律 404。"""
    client = TestClient(app)
    sid = make_session(client)["session_id"]
    assert client.post(f"/sessions/{sid}/end").status_code == 200

    assert client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 0}
    ).status_code == 404
    assert client.post(
        f"/sessions/{sid}/events", json={"time_ms": 0}
    ).status_code == 404
    assert client.post(f"/sessions/{sid}/accept").status_code == 404
    assert client.post(f"/sessions/{sid}/end").status_code == 404


def test_end_snapshot_ignores_unaccepted_scene_revision():
    """快照只读会话自身：未接纳的场景修订不出现在结束快照里。"""
    client = TestClient(app)
    scene_id = "scene-end"
    sid = make_session(client, scene_id)["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    # 场景推进到版本 2，但会话未接纳
    revise(client, scene_id, 1, revised_subtitles_v2())

    resp = client.post(f"/sessions/{sid}/end")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_revision"] == 1
    assert [s["id"] for s in body["subtitles"]] == ["s1", "s2", "s3"]
    assert [s["end_ms"] for s in body["subtitles"]] == [1000, 3000, 2500]

    # 场景本身不受结束影响：仍是版本 2 的内容
    scene = client.get(f"/scenes/{scene_id}").json()
    assert scene["revision"] == 2
    assert [s["id"] for s in scene["subtitles"]] == ["s1", "s2", "s3", "s4"]


def test_end_leaves_scene_and_other_sessions_untouched():
    """结束不改源场景与其他会话；未结束会话的原请求/响应/错误码不变。"""
    client = TestClient(app)
    scene_id = "scene-end"
    first = make_session(client, scene_id)
    second = client.post("/sessions", json={"scene_id": scene_id}).json()
    assert first["session_id"] != second["session_id"]

    assert client.post(f"/sessions/{first['session_id']}/end").status_code == 200

    # 源场景不变
    scene = client.get(f"/scenes/{scene_id}").json()
    assert scene["revision"] == 1
    assert [s["id"] for s in scene["subtitles"]] == ["s1", "s2", "s3"]

    # 另一会话照常推进/事件推进/接纳，响应与错误码不变
    sid2 = second["session_id"]
    body = client.post(f"/sessions/{sid2}/advance", json={"time_ms": 900}).json()
    assert [(a["id"], a["track"]) for a in body["active"]] == [
        ("s1", 1),
        ("s2", 2),
    ]
    assert client.post(
        f"/sessions/{sid2}/advance", json={"time_ms": 899}
    ).status_code == 422
    body = client.post(f"/sessions/{sid2}/events", json={"time_ms": 1000}).json()
    assert [(e["time_ms"], e["type"], e["id"]) for e in body["events"]] == [
        (1000, "exit", "s1"),
        (1000, "enter", "s3"),
    ]
    assert client.post(f"/sessions/{sid2}/accept").status_code == 200

    # 同场景可再建会话；已结束的旧 id 依旧 404
    third = client.post("/sessions", json={"scene_id": scene_id}).json()
    assert third["session_id"] not in (first["session_id"], sid2)
    assert client.post(f"/sessions/{first['session_id']}/end").status_code == 404


def test_end_does_not_alter_previously_returned_snapshots():
    """此前返回的会话快照（创建/接纳响应）不因结束而改变。"""
    client = TestClient(app)
    scene_id = "scene-end"
    created = make_session(client, scene_id)
    sid = created["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})
    accepted = client.post(f"/sessions/{sid}/accept").json()

    ended = client.post(f"/sessions/{sid}/end").json()
    # 结束快照与此前接纳返回的会话状态内容一致（之间无任何会话变更）
    assert ended == accepted
    # 创建响应仍是游标 0 的初始快照，未被结束改写
    assert created["cursor_ms"] == 0
    assert created["source_revision"] == 1
    # 接纳响应保持游标 900 的状态，未被结束改写
    assert accepted["cursor_ms"] == 900
    assert accepted["subtitles"] == created["subtitles"]


def test_end_snapshot_is_pure_data_and_session_content_untouched():
    """manager 级：结束快照逐条复制；结束只置标记，不改会话内容。"""
    manager = PlaybackManager()
    record = SceneRecord(
        scene_id="scene-mgr",
        revision=1,
        track_count=2,
        subtitles=[
            {"id": "s1", "speaker": "A", "start_ms": 0, "end_ms": 1000,
             "locked_track": None},
            {"id": "s2", "speaker": "B", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
        ],
        assignments=[{"id": "s1", "track": 1}, {"id": "s2", "track": 2}],
        changes=0,
    )
    created = manager.create(record)
    session = manager.get(created["session_id"])
    assert session is not None

    snapshot = manager.end(session)
    assert snapshot["session_id"] == created["session_id"]
    # 注册表条目已移除
    assert manager.get(created["session_id"]) is None

    # 结束快照是纯数据副本：改写它不影响会话，也不影响创建时返回的快照
    snapshot["subtitles"][0]["start_ms"] = 999
    snapshot["assignments"][0]["track"] = 9
    assert session.subtitles[0]["start_ms"] == 0
    assert created["subtitles"][0]["start_ms"] == 0
    assert created["assignments"][0]["track"] == 1

    # 会话内容未被结束改动（仅置结束标记）；重复结束抛 SessionEnded
    assert session.state()["subtitles"][0]["start_ms"] == 0
    with pytest.raises(SessionEnded):
        manager.end(session)


# ---------------------------------------------------------------------------
# 可控交错：结束 vs 等待中的推进 / 事件推进 / 接纳 / 重复结束
# ---------------------------------------------------------------------------


def test_end_wins_over_pending_advance(monkeypatch):
    """推进已取得会话引用但尚未生效时结束先生效：在途推进按 404 失败。"""
    client = TestClient(app)
    sid = make_session(client)["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    paused, release = patch_first_manager_call(monkeypatch, "advance")
    responses = {}
    t = post_async(
        f"/sessions/{sid}/advance", responses, "advance", {"time_ms": 1000}
    )
    assert paused.wait(timeout=10), "在途推进未取得会话引用"

    # 推进被暂停期间结束先生效：快照游标仍是 900
    end_resp = client.post(f"/sessions/{sid}/end")
    assert end_resp.status_code == 200
    assert end_resp.json()["cursor_ms"] == 900

    release.set()
    t.join(timeout=10)
    assert not t.is_alive()
    # 在途推进进入会话锁时结束已生效：按未知会话 404，游标未被推进
    assert responses["advance"].status_code == 404
    assert client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 1000}
    ).status_code == 404


def test_end_wins_over_pending_events(monkeypatch):
    """事件推进在途（已取引用未生效）时结束先生效：事件推进按 404 失败。"""
    client = TestClient(app)
    sid = make_session(client)["session_id"]

    paused, release = patch_first_manager_call(monkeypatch, "advance_events")
    responses = {}
    t = post_async(
        f"/sessions/{sid}/events", responses, "events", {"time_ms": 3000}
    )
    assert paused.wait(timeout=10), "在途事件推进未取得会话引用"

    end_resp = client.post(f"/sessions/{sid}/end")
    assert end_resp.status_code == 200
    assert end_resp.json()["cursor_ms"] == 0

    release.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert responses["events"].status_code == 404
    assert client.post(
        f"/sessions/{sid}/events", json={"time_ms": 3000}
    ).status_code == 404


def test_end_wins_over_pending_accept(monkeypatch):
    """接纳在途（已取引用未生效）时结束先生效：接纳 404，不会复活会话。"""
    client = TestClient(app)
    scene_id = "scene-end"
    sid = make_session(client, scene_id)["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})
    revise(client, scene_id, 1, revised_subtitles_v2())

    paused, release = patch_first_manager_call(monkeypatch, "accept")
    responses = {}
    t = post_async(f"/sessions/{sid}/accept", responses, "accept")
    assert paused.wait(timeout=10), "在途接纳未取得会话引用"

    # 接纳尚未生效，结束快照仍是版本 1 的会话状态
    end_resp = client.post(f"/sessions/{sid}/end")
    assert end_resp.status_code == 200
    assert end_resp.json()["source_revision"] == 1

    release.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert responses["accept"].status_code == 404
    # 会话已释放：后续接纳/结束同样 404
    assert client.post(f"/sessions/{sid}/accept").status_code == 404
    assert client.post(f"/sessions/{sid}/end").status_code == 404


def test_duplicate_end_only_one_succeeds(monkeypatch):
    """重复结束的可控交错：先到者取走快照，迟到者 404。"""
    client = TestClient(app)
    sid = make_session(client)["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    # 第一次结束在取得会话引用后、生效前被暂停；第二次结束先生效
    paused, release = patch_first_manager_call(monkeypatch, "end")
    responses = {}
    t1 = post_async(f"/sessions/{sid}/end", responses, "first")
    assert paused.wait(timeout=10), "第一次结束未取得会话引用"

    second = client.post(f"/sessions/{sid}/end")
    assert second.status_code == 200
    assert second.json()["cursor_ms"] == 900

    release.set()
    t1.join(timeout=10)
    assert not t1.is_alive()
    # 迟到的重复结束：进入会话锁时结束已生效，按 404 失败
    assert responses["first"].status_code == 404
    assert client.post(f"/sessions/{sid}/end").status_code == 404


def test_concurrent_ends_exactly_one_success():
    """并发结束：恰好一次 200（完整快照），其余全部 404。"""
    client = TestClient(app)
    sid = make_session(client)["session_id"]

    results = []
    barrier = threading.Barrier(6)

    def worker():
        local = TestClient(app)
        barrier.wait()
        resp = local.post(f"/sessions/{sid}/end")
        results.append((resp.status_code, resp.json()))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(status for status, _ in results)
    assert statuses == [200, 404, 404, 404, 404, 404]
    winner = next(body for status, body in results if status == 200)
    assert winner["session_id"] == sid
    assert winner["cursor_ms"] == 0
    assert [s["id"] for s in winner["subtitles"]] == ["s1", "s2", "s3"]
    assert winner["assignments"] == [
        {"id": "s1", "track": 1},
        {"id": "s2", "track": 2},
        {"id": "s3", "track": 1},
    ]
    # 会话已释放
    assert client.post(f"/sessions/{sid}/end").status_code == 404


def test_end_waits_for_inflight_accept_and_snapshots_accepted_state(monkeypatch):
    """结束与接纳串行：接纳持锁期间结束不得插队，快照取自接纳后的会话。"""
    client = TestClient(app)
    scene_id = "scene-end"
    sid = make_session(client, scene_id)["session_id"]
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})

    # 场景推进到版本 2（s3 延长到 4000，新增 s4）
    revise(client, scene_id, 1, revised_subtitles_v2())

    # 接纳在会话锁内读取场景时被暂停：会话锁由该接纳请求持有
    paused, release = patch_first_scene_read(monkeypatch)
    responses = {}
    t1 = post_async(f"/sessions/{sid}/accept", responses, "accept")
    assert paused.wait(timeout=10), "接纳未在会话锁内读取场景"

    # 结束与接纳串行：接纳的完整操作生效前，结束不得生效
    t2 = post_async(f"/sessions/{sid}/end", responses, "end")
    try:
        t2.join(timeout=0.5)
        assert t2.is_alive(), "结束在接纳的完整操作生效前插队"
    finally:
        release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive() and not t2.is_alive()

    assert responses["accept"].status_code == 200
    end_resp = responses["end"]
    assert end_resp.status_code == 200
    body = end_resp.json()
    # 快照取自轮到结束生效时（即接纳之后）的会话：版本 2 的内容
    assert body["source_revision"] == 2
    assert body["cursor_ms"] == 900
    assert [s["id"] for s in body["subtitles"]] == ["s1", "s2", "s3", "s4"]
    assert body["assignments"] == [
        {"id": "s1", "track": 1},
        {"id": "s2", "track": 2},
        {"id": "s3", "track": 1},
        {"id": "s4", "track": 2},
    ]
    # 会话已释放
    assert client.post(f"/sessions/{sid}/end").status_code == 404
