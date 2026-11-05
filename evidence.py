"""证据契约（本项目的灵魂模块）。

判定"回答里出现的价格是否都有依据"：
允许的价格只能来自 ① 画面感知（价签/读出的文字） ② 工具返回结果（比价等）。
凡是回答中出现、却不在允许集合里的价格，都记为"无证据断言"（violation）。
该模块同时被运行时（finalize 节点）与评测（evals/）复用。
"""
import re

_PRICE_TOKEN = re.compile(r"(?:¥|￥)\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*元")


def _norm(price: str) -> str:
    """统一价格格式便于集合比较：剥离货币符号等非数字字符，'49.90'->'49.9'。

    真实 VLM 可能返回 '¥49.90' / '￥49.90 元' 等带符号写法，一律归一化。
    """
    m = re.search(r"\d+(?:\.\d{1,2})?", str(price))
    if not m:
        return str(price)
    return ("%.2f" % float(m.group())).rstrip("0").rstrip(".") or "0"


def extract_prices(text: str) -> list:
    """提取文本中出现的所有价格数字（含 ¥xx / xx元 两种写法）。"""
    out = []
    for m in _PRICE_TOKEN.finditer(text or ""):
        val = m.group(1) or m.group(2)
        if val:
            out.append(val)
    return out


# 中文数字（证据原文常写"不到十块/十几元"），回答里被转写成阿拉伯数字时不应算违规
_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_NUM = re.compile(r"([零一二两三四五六七八九十百千]{1,6})\s*(?:元|块)")


def _cn_to_int(s: str) -> int:
    section, num = 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        else:
            section += (num or 1) * _CN_UNIT[ch]
            num = 0
    return section + num


def cn_prices(text: str) -> list:
    """提取"十块 / 二十五元 / 一百二十块"这类中文数字价格。"""
    out = []
    for m in _CN_NUM.finditer(text or ""):
        v = _cn_to_int(m.group(1))
        if 0 < v <= 100000:
            out.append(str(v))
    return out


def allowed_prices(state: dict) -> set:
    """从状态里收集"有依据"的价格集合：
    感知 items + 感知 read_text + 工具结果（标题/摘要，含中文数字写法）。"""
    allowed = set()
    perception = state.get("perception") or {}
    for item in perception.get("items") or []:
        if item.get("price"):
            allowed.add(_norm(item["price"]))

    texts = list(perception.get("read_text") or [])
    for tr in state.get("tool_results") or []:
        out = tr.get("output") or {}
        # 画像事实（query_profile 返回 {"facts": [...]}）是用户【自己说过的话】，
        # 属于有出处的信息：漏掉它会把"我预算 200 元"误判成无证据断言。
        texts.extend(str(f) for f in (out.get("facts") or []))
        for r in out.get("results") or []:
            if r.get("price"):
                allowed.add(_norm(r["price"]))
            # text 是检索类工具（search_kb）的正文片段：来自记忆/知识库，
            # 与比价结果的 title 一样属于"有出处"的价格来源
            texts.append(r.get("title") or "")
            texts.append(r.get("note") or "")
            texts.append(r.get("text") or "")

    for text in texts:
        for p in extract_prices(text) + cn_prices(text):
            allowed.add(_norm(p))
    return allowed


def uses_demo_data(state: dict) -> bool:
    """本轮工具结果里是否存在演示（mock）数据。"""
    for tr in state.get("tool_results") or []:
        ev = (tr.get("output") or {}).get("evidence") or {}
        if ev.get("is_demo"):
            return True
    return False


# 引用编号：【记忆:chunk_id】或 [记忆:chunk_id]（模型两种写法都出现过）
_CITE_RE = re.compile(r"【?记忆[:：]([^】\]]+)】?")


def cited_ids(answer: str) -> list:
    """抽出回答里引用的知识库编号。"""
    return [c.strip() for c in _CITE_RE.findall(answer or "") if c.strip()]


def check_citations(answer: str, state: dict) -> list:
    """引用 grounding 校验：引用的编号必须真的来自本轮检索结果。

    这是证据契约在 RAG 上的延伸——光"给了引用"还不够，引用还得是真的。
    编造一个没检索到的编号，和编造一个价格一样属于幻觉。
    """
    valid = set()
    for tr in state.get("tool_results") or []:
        if tr.get("tool") != "search_kb":
            continue
        for r in (tr.get("output") or {}).get("results") or []:
            if r.get("chunk_id"):
                valid.add(r["chunk_id"])
    return [{"type": "引用不存在", "chunk_id": c}
            for c in cited_ids(answer) if c not in valid]


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def derived_prices(allowed: set) -> set:
    """"有依据的计算"：由已允许的价格做【受限】运算，得到可推导的价格。

    现实场景：画面 49.90 + 促销"第二件半价" → 回答算出"两只共 74.85、
    折合 37.4 一只"。这些数字没有直接出现在证据里，但完全可由证据推导，
    判成幻觉是误伤（真实链路已踩到）。

    为了不把"随便编一个数"也放行，只允许三类受限推导，且最多两层：
      ① 折扣：a × {0.5, 0.8, 0.9}
      ② 均摊：a ÷ {2, 3, 4}
      ③ 合计：任意两个允许价格相加；以及"合计"再除以 2 / 3
    这样能覆盖"半价合计 → 再均摊"，同时集合规模保持在千级、可解释。
    """
    # 只取前 12 个价格参与推导，避免工具结果多时集合爆炸
    base = sorted({v for v in (_to_float(x) for x in allowed) if v and v > 0})[:12]
    if not base:
        return set()

    # 第一层：折扣与均摊（49.90 × 0.5 = 24.95）
    level1 = set(base)
    for a in base:
        for f in (0.5, 0.8, 0.9):
            level1.add(round(a * f, 2))
        for n in (2, 3, 4):
            level1.add(round(a / n, 2))

    # 第二层：合计 = 任意两个"一层值"相加（49.90 + 24.95 = 74.85）
    out = set(level1)
    ordered = sorted(level1)
    for i, a in enumerate(ordered):
        for b in ordered[i:]:
            out.add(round(a + b, 2))

    # 第三层：合计再均摊（74.85 ÷ 2 = 37.425 → 37.4）
    for a in set(out):
        for n in (2, 3):
            out.add(round(a / n, 2))
    return out


def _covered(value: float, candidates: set) -> bool:
    """容差匹配：计算结果通常被四舍五入（37.425 → 37.4、74.85 → 74.8）。"""
    for c in candidates:
        if abs(value - c) <= max(0.05, c * 0.005):
            return True
    return False


def check_answer(answer: str, state: dict) -> dict:
    """校验最终回答：返回违规清单与明细（供展示与评测）。

    价格判定分三步，前两步都算"有依据"：
      ① 精确命中：数字直接出现在证据里；
      ② 可推导命中：由证据价格经受限运算得到（促销合计、均摊、折扣）；
      ③ 都不命中 → 判为"无证据价格"。
    """
    allowed = allowed_prices(state)
    allowed_f = {v for v in (_to_float(x) for x in allowed) if v is not None}
    derived = derived_prices(allowed)
    violations = []
    for p in extract_prices(answer or ""):
        v = _to_float(p)
        if v is None:
            continue
        if v in allowed_f or _covered(v, derived):
            continue
        violations.append({"type": "无证据价格", "price": p})
    if uses_demo_data(state) and "演示" not in (answer or ""):
        violations.append({"type": "未声明演示数据"})
    violations.extend(check_citations(answer, state))
    return {
        "violations": violations,
        "price_claims": extract_prices(answer or ""),
        "allowed_prices": sorted(allowed),
        "derived_prices": sorted(round(x, 2) for x in derived),
        "citations": cited_ids(answer),
    }
