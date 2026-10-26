"""流式输出事件通道（可选能力，默认关闭）。

设计要点：
- 用 ContextVar 传递发射器，节点里拿不到就退化成一次性输出，
  因此 CLI / 评测 / 测试完全不受影响（结果与流式路径逐字相同）；
- 只有在 Web 流式接口里才 bind 发射器，图结构、提示词、工具、证据校验
  全部复用同一套代码，保证「流式结果 == 一次性结果」。

事件类型（SSE 帧 data 里的 JSON）：
    start       开始标识：{"request_id", "modes"}
    phase       阶段进度：{"stage", "label"}
    perception  画面感知：{"data"}
    tool        工具结果：{"name", "output"}
    links       候选链接：{"items":[{platform,title,price,url}]}
    delta       正文增量：{"text"}
    rollback    撤回本轮增量（该轮最终发起了工具调用）
    violations  证据校验：{"items"}
    done        结束标识：{"answer","perception","tool_results","violations","links","modes"}
    error       异常结束：{"message","stage"}
    （另有注释帧 ": ping" 作为心跳，非 JSON）
"""
import contextvars
import queue

_current = contextvars.ContextVar("snap_stream_emitter", default=None)


def current():
    """获取当前线程绑定的发射器（未绑定返回 None）。"""
    return _current.get()


def bind(emitter):
    return _current.set(emitter)


def unbind(token):
    _current.reset(token)


class Emitter:
    """把事件放入队列，由 SSE 生成器取出推送。

    stop 标志用于客户端断开后停止入队，避免队列无限增长；
    工作线程会自然跑完（不会硬杀正在进行的模型调用）。
    """

    END = "__end__"

    def __init__(self):
        self.queue = queue.Queue()
        self.stopped = False
        self.last_links_sig = None  # 链接集合签名：多轮工具时不重复推送同一批链接

    def emit(self, type_: str, **payload):
        if self.stopped:
            return
        payload["type"] = type_
        self.queue.put(payload)

    def end(self):
        """发送流结束哨兵。"""
        self.queue.put(self.END)

    def stop(self):
        """客户端断开/超时后调用：后续事件不再入队。"""
        self.stopped = True
