"""FastAPI 网页版：上传照片 + 提问 → 感知 / 工具 / 证据约束回答。

启动：python main.py --web  →  http://127.0.0.1:8000

接口：
  GET  /api/modes        当前各层模式（mock/real）
  GET  /api/sample       示例货架图
  POST /api/ask          一次性提问（multipart：question 必填，image 可选）
  POST /api/ask/stream   流式提问（SSE，事件协议见 README「流式输出」章节）
  GET  /api/history      最近对话存档

流式与一次性共用同一张 LangGraph、同一套工具与证据校验，
返回的字段完全一致（answer / perception / tool_results / violations / links）。
"""
import asyncio
import json
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from agent import runner, streaming
from memory import store

app = FastAPI(title="小拍 · 拍照购物参谋")

# 心跳间隔（秒）：长时间无数据时发注释帧，用于穿透反代缓冲、维持长连接
HEARTBEAT_INTERVAL = 10
# 队列轮询间隔（秒）：避免阻塞线程，也便于随时响应客户端断开
POLL_INTERVAL = 0.05
# 单次请求硬超时（秒）：超过则主动发 error 结束流，不让客户端无限等待
HARD_TIMEOUT = 180


@app.get("/api/modes")
def modes():
    return config.mode_summary()


@app.get("/api/sample")
def sample():
    runner.ensure_sample()
    return FileResponse(str(config.SAMPLE_IMAGE), media_type="image/jpeg")


@app.post("/api/ask")
async def ask(question: str = Form(...), image: UploadFile | None = File(None)):
    image_path = await _save_upload(image)
    result = runner.run_once(question, image_path)
    return _payload(result)


@app.post("/api/ask/stream")
async def ask_stream(question: str = Form(...), image: UploadFile | None = File(None)):
    """流式提问：SSE 逐块推送，字段与 /api/ask 完全一致。

    图在后台线程里跑（LLM/VLM 是同步 SDK），事件通过队列交给协程推送；
    客户端断开时只停止入队，不做硬杀，避免半截状态。
    """
    image_path = await _save_upload(image)
    emitter = streaming.Emitter()
    request_id = uuid.uuid4().hex[:8]
    state = {"closed": False, "timer": None}

    def _close(type_: str, **payload):
        """发出且只发出一个结束标识（done / error），随后关流。

        这是"客户端绝不会永远等下去"的第一道保证：无论成功、异常、还是
        工作线程半路退出，SSE 流一定会以 done 或 error 收尾。
        """
        if state["closed"]:
            return
        state["closed"] = True
        if state["timer"]:
            state["timer"].cancel()
        emitter.emit(type_, **payload)
        emitter.end()

    def worker():
        token = streaming.bind(emitter)
        try:
            result = runner.run_once(question, image_path)
            _close("done", **_payload(result))
        except Exception as e:  # noqa: BLE001
            _close("error", message=str(e)[:300], stage="graph")
        finally:
            # 兜底：若上面两条路径都没走到（如 BaseException），也必须给结束标识
            _close("error", message="服务端异常退出，未生成结束标识", stage="worker")
            streaming.unbind(token)

    def watchdog():
        """硬超时：模型调用若挂死，主动发 error 结束流，前端立刻可知。"""
        _close("error", stage="timeout",
               message="处理超时（超过 %d 秒未完成），已中止等待，请重试" % HARD_TIMEOUT)

    threading.Thread(target=worker, daemon=True).start()
    timer = threading.Timer(HARD_TIMEOUT, watchdog)
    timer.daemon = True
    state["timer"] = timer
    timer.start()
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # 关闭 nginx 等反代的响应缓冲
    }
    return StreamingResponse(_sse(emitter, request_id),
                            media_type="text/event-stream", headers=headers)


@app.get("/api/history")
def history():
    return store.recent_turns(20)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

async def _save_upload(image) -> str:
    """保存上传图片；未上传时回退示例图，保证零照片也能体验完整链路。"""
    if image and image.filename:
        config.ensure_dirs()
        safe_name = "%s_%s" % (uuid.uuid4().hex[:12], Path(image.filename).name)
        path = config.UPLOAD_DIR / safe_name
        path.write_bytes(await image.read())
        return str(path)
    if not config.SAMPLE_IMAGE.exists():
        runner.ensure_sample()
    return str(config.SAMPLE_IMAGE)


def _payload(result: dict) -> dict:
    """一次性接口与流式 done 事件共用的响应结构。"""
    return {
        "answer": result.get("final_answer"),
        "perception": result.get("perception"),
        "tool_results": result.get("tool_results", []),
        "violations": result.get("violations", []),
        "links": result.get("links", []),
        "modes": config.mode_summary(),
    }


async def _sse(emitter, request_id: str):
    """把队列里的事件逐条编码成 SSE 帧。"""
    yield _frame({"type": "start", "request_id": request_id,
                  "modes": config.mode_summary()})
    last = time.monotonic()
    try:
        while True:
            try:
                item = emitter.queue.get_nowait()
            except queue.Empty:
                # 队列为空：到点发一次心跳（注释帧，客户端忽略即可）
                if time.monotonic() - last >= HEARTBEAT_INTERVAL:
                    yield ": ping\n\n"
                    last = time.monotonic()
                await asyncio.sleep(POLL_INTERVAL)
                continue
            if item == streaming.Emitter.END:
                break
            yield _frame(item)
            last = time.monotonic()
    finally:
        # 客户端断开/超时后停止入队（工作线程会自然跑完，不做硬杀）
        emitter.stop()


def _frame(obj: dict) -> str:
    return "data: %s\n\n" % json.dumps(obj, ensure_ascii=False)


# 前端静态页（挂在根路径）
app.mount("/", StaticFiles(directory=str(config.ROOT / "app" / "static"), html=True),
          name="static")
