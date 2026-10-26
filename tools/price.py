"""比价工具：一切结果都遵循 results + evidence 契约。

PRICE_MODE=mock（默认）：返回明确标注"演示数据"的固定报价，保证零 Key 离线可跑；
PRICE_MODE=web：Tavily（可选，需 Key）→ DuckDuckGo → Bing 逐级回退的网页检索，
               从标题/摘要中抽取价格。

设计原则（诚实 Agent）：
- 绝不做"静默 mock 回退"——查不到就返回空结果并说明原因，而不是假装查到了；
- 每个结果都带 source / fetched_at / is_demo，供回答约束与评测使用。
"""
import re
import time

import requests

import config

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9",
}
PRICE_RE = re.compile(r"(?:¥|￥)\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*元")

# 电商站点白名单：Tavily 定向检索，避免"森马"这类品牌名撞上同名上市公司新闻
_SHOP_DOMAINS = ["jd.com", "tmall.com", "taobao.com", "pinduoduo.com", "suning.com",
                 "smzdm.com", "vip.com", "kaola.com", "amazon.cn"]
# 检索噪声词：VLM 给的营销长标题里的修饰语，剔除后更容易命中真实商品页
_NOISE_WORDS = ("新款", "秋季", "夏季", "春季", "冬季", "女鞋", "男鞋", "女款", "男款",
                "女士", "男士", "休闲", "运动", "正品", "官方", "旗舰店", "包邮",
                "学生", "百搭", "厚底", "增高", "同款", "爆款", "热卖", "专柜")
# 财经/新闻类结果与商品比价无关，直接丢弃
_IRRELEVANT_HINTS = ("股价", "股票", "涨停", "跌停", "财报", "基金", "证券", "市值",
                     "融资", "净值", "stock", "finance", "news")


def _core_query(product: str) -> str:
    """把营销长标题提纯为检索核心词，如
    '森马小白鞋女鞋秋季新款休闲运动板鞋' -> '森马 小白鞋'。"""
    q = re.sub(r"[（）()【】\[\]「」]", " ", product or "")
    for w in _NOISE_WORDS:
        q = q.replace(w, " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q or (product or "")


def _core_tokens(product: str) -> list:
    parts = re.split(r"[\s,，、/]+", _core_query(product))
    return [p for p in parts if len(p) >= 2][:4]


def _relevant(product: str, results: list) -> list:
    """相关性过滤：丢弃财经新闻类页面，以及标题/摘要不含任何核心词的页面。
    这是比价可用性的关键——宁可返回空（如实说查不到），也不喂给模型垃圾。"""
    toks = _core_tokens(product)
    keep = []
    for r in results:
        text = ("%s %s %s" % (r.get("title", ""), r.get("note", ""),
                              r.get("url", ""))).lower()
        if any(h in text for h in _IRRELEVANT_HINTS):
            continue
        if toks and not any(t.lower() in text for t in toks):
            continue
        keep.append(r)
    return keep

# mock 模式的演示商品库（结果全部明确标注"演示数据"）
_KNOWN_MOCK = {
    "保温杯": [("京东", "不锈钢真空保温杯 500ml", "45.00"),
               ("淘宝", "保温杯官方旗舰店", "42.50"),
               ("拼多多", "保温杯 百亿补贴", "39.90")],
    "玻璃水杯": [("京东", "双层玻璃水杯 350ml", "15.90"),
                 ("淘宝", "玻璃水杯 家用透明", "12.80")],
    "不锈钢饭盒": [("京东", "不锈钢保温饭盒", "38.00"),
                   ("淘宝", "不锈钢分格饭盒", "33.50")],
    "马克杯": [("京东", "陶瓷马克杯 380ml", "22.00"),
               ("淘宝", "马克杯 简约家用", "18.90")],
}


def _evidence(source: str, query: str, **extra) -> dict:
    ev = {"source": source, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
          "query": query, "is_demo": source == "mock"}
    ev.update(extra)
    return ev


def _resp(query, results, source, reason=None, **extra) -> dict:
    ev = _evidence(source, query, **extra)
    if reason:
        ev["reason"] = reason
    return {"query": query, "results": results, "evidence": ev}


def _mock_results(product: str) -> list:
    for key, offers in _KNOWN_MOCK.items():
        if key in product:
            return [{"platform": p, "title": t, "price": pr, "url": "",
                     "note": "演示数据"}
                    for p, t, pr in offers]
    return [{"platform": p, "title": "%s（演示数据）" % product, "price": pr, "url": "",
             "note": "演示数据"}
            for p, pr in (("京东", "29.90"), ("淘宝", "25.50"), ("拼多多", "19.90"))]


def search_price_offers(product: str) -> dict:
    """入口：返回 {query, results, evidence} 契约，绝不静默造假。"""
    product = (product or "").strip()
    if not product:
        return _resp("", [], "mock", "empty_product")
    if config.PRICE_MODE == "mock":
        return _resp(product, _mock_results(product), "mock")

    # web 模式：Tavily → DuckDuckGo → Bing 逐级回退，结果统一做相关性过滤
    for name, provider in (("tavily", _tavily), ("duckduckgo", _ddg), ("bing", _bing)):
        try:
            raw = provider(product) or []
        except Exception:
            continue
        results = _relevant(product, raw)
        if results:
            results = _enrich_prices(results[:5])
            # 有价格的排前面：模型优先看到可引用的报价
            results.sort(key=_price_sort_key)
            priced = [r for r in results if r.get("price")]
            return _resp(product, results, "web", engine=name,
                         core_query=_core_query(product),
                         dropped=len(raw) - len(results),
                         priced=len(priced))
    return _resp(product, [], "web", "no_reliable_result",
                 core_query=_core_query(product))


# ---------------------------------------------------------------------------
# 价格补采：电商列表页/搜索结果里价格常由 JS 渲染，正文抽不到；
# 对京东商品详情页做一次轻量回抓，能拿到就补上（拿不到保持 None，不编造）。
# ---------------------------------------------------------------------------

_JD_PRICE_PATTERNS = (
    r'"p"\s*:\s*"([\d.]+)"',                                   # 京东详情页价格字段
    r'class="price[^"]*"[^>]*>\s*(?:<[^>]+>\s*)*([\d.]+)',      # 旧版价格标签
)


def _jd_price(html: str):
    for pat in _JD_PRICE_PATTERNS:
        m = re.search(pat, html or "")
        if not m:
            continue
        try:
            v = float(m.group(1))
        except (TypeError, ValueError):
            continue
        if 1 <= v <= 100000:          # 过滤掉时间戳/ID 之类的伪价格
            return "%.2f" % v
    return None


def _enrich_prices(results: list, max_pages: int = 2) -> list:
    fetched = 0
    for r in results:
        if fetched >= max_pages or r.get("price"):
            continue
        if not re.search(r"item(\.m)?\.jd\.com/", r.get("url") or ""):
            continue
        try:
            html = requests.get(r["url"], headers=HEADERS, timeout=6).text
        except Exception:
            continue
        p = _jd_price(html)
        if p:
            r["price"] = p
            r["note"] = ((r.get("note") or "")[:80] + "（价格取自商品页）").strip()
            fetched += 1
    return results


# ---------------------------------------------------------------------------
# web 模式的三个检索后端（无 Key 时 DuckDuckGo / Bing 兜底）
# ---------------------------------------------------------------------------

def _extract_price(text: str):
    m = PRICE_RE.search(text or "")
    return (m.group(1) or m.group(2)) if m else None


def _price_sort_key(r: dict):
    """有价格的排前面，并按价格升序；无价格的排在最后。"""
    p = r.get("price")
    try:
        return (0, float(p))
    except (TypeError, ValueError):
        return (1, 0.0)


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "")


def _tavily(query: str, limit=5):
    if not config.TAVILY_API_KEY:
        return None
    q = "%s 价格" % _core_query(query)
    base = {"api_key": config.TAVILY_API_KEY, "query": q, "max_results": 8,
            "search_depth": "basic"}
    out = []
    # 第一轮定向电商站点（命中率最高）；第二轮放开全网（站点白名单可能漏掉比价博客）
    for extra in ({"include_domains": _SHOP_DOMAINS}, {}):
        payload = dict(base)
        payload.update(extra)
        try:
            resp = requests.post("https://api.tavily.com/search", json=payload,
                                 headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception:
            continue
        for item in (resp.json().get("results") or []):
            title = (item.get("title") or "").strip()
            url = item.get("url") or ""
            snippet = (item.get("content") or "").strip()
            if not title:
                continue
            tld = url.split("/")[2] if "://" in url and len(url.split("/")) > 2 else ""
            out.append({"platform": tld or "网页搜索", "title": title,
                        "price": _extract_price(snippet) or _extract_price(title),
                        "url": url, "note": snippet[:120]})
            if len(out) >= limit * 2:
                break
        if out:
            break
    return out or None


def _ddg(query: str, limit=3):
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": "%s 价格" % _core_query(query)}, headers=HEADERS,
        timeout=config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for block in re.split(r'<div[^>]*class="result[^"]*"', resp.text)[1:]:
        m = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        title = _strip_tags(m.group(2)).strip()
        if not title:
            continue
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</', block, re.S)
        snippet = _strip_tags(sm.group(1)).strip() if sm else ""
        out.append({"platform": "网页搜索(DuckDuckGo)", "title": title,
                    "price": _extract_price(snippet) or _extract_price(title),
                    "url": m.group(1), "note": snippet[:120]})
        if len(out) >= limit:
            break
    return out or None


def _bing(query: str, limit=3):
    resp = requests.get(
        "https://www.bing.com/search",
        params={"q": "%s 价格" % _core_query(query), "setlang": "zh-CN", "mkt": "zh-CN"},
        headers=HEADERS, timeout=config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for block in re.findall(r'<li class="b_algo".*?</li>', resp.text, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        title = _strip_tags(m.group(2)).strip()
        if not title:
            continue
        pm = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        snippet = _strip_tags(pm.group(1)).strip() if pm else ""
        out.append({"platform": "网页搜索(Bing)", "title": title,
                    "price": _extract_price(snippet) or _extract_price(title),
                    "url": m.group(1), "note": snippet[:120]})
        if len(out) >= limit:
            break
    return out or None
