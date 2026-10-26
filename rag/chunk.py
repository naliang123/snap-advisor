"""文本切分：把长文本切成可检索的小块。

为什么需要切分（这是 RAG 里最容易被忽略、却最影响效果的一步）：
  - 整篇塞进去 → 向量被无关内容稀释，相关度算不准；
  - 切得太碎 → 一块里没有完整语义，答非所问；
  - 带重叠（overlap）→ 避免关键信息正好被切在两块中间。

切多大合适没有标准答案，所以 evals/rag_ablation.py 里做了 200/500/1000 字的
对比实验，用召回率说话，而不是拍脑袋。
"""
import re

DEFAULT_SIZE = 300      # 每块目标字数
DEFAULT_OVERLAP = 60    # 相邻块重叠字数

# 优先在这些标点后断开，尽量不把一句话切成两半
_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def split_text(text: str, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP):
    """切分文本：先按标点断句，再贪心合并到 size；相邻块保留 overlap 字符。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # 1) 按标点切成片段
    pieces = [p for p in _SPLIT_RE.split(text) if p]
    # 2) 超长片段（没有标点）硬切，避免单块过大
    flat = []
    for p in pieces:
        while len(p) > size:
            flat.append(p[:size])
            p = p[size:]
        if p:
            flat.append(p)

    # 3) 贪心合并
    chunks, cur = [], ""
    for p in flat:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur.strip())
            cur = (cur[-overlap:] if overlap else "") + p
        else:
            cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]
