"""文本向量：真实（OpenAI 兼容 /embeddings）与离线 Mock 双模式。

设计原则与 llm/client.py 一致：零 Key 也能跑通全链路，且结果可复现。

【必须说清的一点（诚实原则）】
mock 嵌入是【词面哈希伪向量】：把汉字单字/bigram 与英文词哈希到固定维度，
衡量的是"字面重合度"，**不具备语义泛化能力**——"便宜"和"实惠"在 mock 下
不会被判为相似。只有真实模式（配置 EMBED_API_KEY）才是语义 embedding。

两种向量维度不同，因此库中按 embed_mode + dim 隔离存储，检索时只比对同模式
向量，绝不把伪向量和真向量混在一起算相似度。
"""
import hashlib
import math
import re
import struct

import config

# mock 向量维度：256 维足够放下常用汉字 bigram 的哈希空间，纯 Python 计算毫秒级
MOCK_DIM = 256
# 真实模式批量上限（不同供应商上限不同，保守取 32）
REAL_BATCH = 32


def _l2(vec):
    s = math.sqrt(sum(x * x for x in vec))
    if s <= 0:
        return vec
    return [x / s for x in vec]


def _tokens(text: str):
    """分词：英文/数字按词；中文按【单字 + bigram】（bigram 承载词面重合）。"""
    t = (text or "").lower()
    toks = re.findall(r"[a-z0-9]+", t)
    for seg in re.findall(r"[\u4e00-\u9fff]+", t):
        toks.extend(list(seg))
        toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


def mock_embed(text: str):
    """离线确定性伪嵌入：同样的文本永远得到同样的向量（评测可复现）。"""
    vec = [0.0] * MOCK_DIM
    for tk in _tokens(text):
        h = int(hashlib.md5(tk.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % MOCK_DIM] += 1.0
    return _l2(vec)


def _real_embed(texts):
    from openai import OpenAI
    client = OpenAI(api_key=config.EMBED_API_KEY, base_url=config.EMBED_BASE_URL)
    out = []
    for i in range(0, len(texts), REAL_BATCH):
        resp = client.embeddings.create(
            model=config.EMBED_MODEL, input=texts[i:i + REAL_BATCH])
        out.extend(_l2([float(x) for x in d.embedding]) for d in resp.data)
    return out


def embed_many(texts):
    texts = list(texts or [])
    if not texts:
        return []
    if config.EMBED_MODE == "real":
        return _real_embed(texts)
    return [mock_embed(t) for t in texts]


def embed_one(text: str):
    vecs = embed_many([text])
    return vecs[0] if vecs else []


def probe_dim() -> int:
    """当前模式下向量的实际维度（建库时记录，用于隔离不同模式的向量）。"""
    v = embed_one("维度探测")
    return len(v)


def pack(vec) -> bytes:
    return struct.pack("<%df" % len(vec), *vec)


def unpack(blob: bytes, dim: int = None):
    if not blob:
        return []
    n = dim or (len(blob) // 4)
    return list(struct.unpack("<%df" % n, blob))


def cosine(a, b) -> float:
    """余弦相似度（约定入参已 L2 归一化，故退化为点积）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
