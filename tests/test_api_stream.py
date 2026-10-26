"""Web 接口层：流式（SSE）与一次性接口返回内容一致。

验证点：
- 响应头符合流式要求（text/event-stream + 关闭反代缓冲）；
- 帧格式可解析，以 start 开始、done 结束；
- done 与 /api/ask 的字段完全相同（流式不改变任何结果）；
- 逐块内容拼接后是最终答案的正文部分。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import server

QUESTION = "保温杯值不值？帮我比价"


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


def _parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or not block.startswith("data:"):
            continue  # 心跳是注释帧（: ping），这里跳过
        events.append(json.loads(block[5:].strip()))
    return events


def test_stream_headers(client):
    resp = client.post("/api/ask/stream", data={"question": QUESTION})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers.get("x-accel-buffering") == "no"
    assert resp.headers.get("cache-control") == "no-cache, no-transform"


def test_stream_matches_once_result(client):
    once = client.post("/api/ask", data={"question": QUESTION}).json()
    events = _parse_sse(
        client.post("/api/ask/stream", data={"question": QUESTION}).text)

    types = [e["type"] for e in events]
    assert types[0] == "start", "第一帧必须是开始标识"
    assert types[-1] == "done", "最后一帧必须是结束标识"

    done = events[-1]
    for key in ("answer", "perception", "tool_results", "violations", "links"):
        assert done[key] == once[key], "流式与一次性结果必须一致：%s" % key

    # 逐块推送的正文拼接后，是最终答案的正文（链接块由服务端确定性补全）
    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas
    assert done["answer"].startswith(deltas)


def test_stream_emits_progress_events(client):
    events = _parse_sse(
        client.post("/api/ask/stream", data={"question": QUESTION}).text)
    types = [e["type"] for e in events]
    for expected in ("start", "phase", "perception", "delta", "done"):
        assert expected in types
    # 阶段事件里应能看出"看图/组织回答/比价"的进度
    labels = [e.get("label", "") for e in events if e["type"] == "phase"]
    assert any("图" in x for x in labels)


def test_status_bar_can_actually_hide(client):
    """回归保护：状态条靠 hidden 属性隐藏，而它的作者样式是 display:flex，
    会压过 UA 的 [hidden]{display:none}——必须显式声明 .status[hidden]。

    这个坑曾让状态条永久停在"正在组织回答…"，被误认为流程卡死一个多小时。
    """
    html = client.get("/").text
    assert 'id="status"' in html and "hidden" in html
    assert ".status[hidden]" in html
    # 声明的应该是"隐藏"，不是别的
    idx = html.index(".status[hidden]")
    assert "display:none" in html[idx:idx + 60].replace(" ", "")
