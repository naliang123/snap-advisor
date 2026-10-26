"""VLM 感知层：把一张商品/货架照片变成结构化 JSON 观察。

真实模式：任意 OpenAI 兼容多模态接口（默认通义 qwen-vl-plus）；
Mock 模式：读取 data/perceptions/<图片名>.json 标注文件
           （与 scripts/make_sample_image.py 配套生成），零 Key 可跑。

感知 JSON 契约（evidence 契约的起点）：
{
  "scene": "一句话场景描述",
  "items": [{"name", "price", "promo", "count", "confidence"}],  # price 仅来自画面价签
  "read_text": ["画面其他可读文字"],
  "uncertain": ["看不清/拿不准的内容"]
}
"""
import base64
import json
from pathlib import Path

import config

PERCEPTION_PROMPT = """你是商品视觉感知器。仔细看这张照片，只报告画面里【真实可见】的信息。
严格只输出一个 JSON（不要输出任何其他文字）：
{
  "scene": "一句话描述场景",
  "items": [{"name": "商品名", "price": "价签上的纯数字字符串（不要包含¥等货币符号），没看到价签则为 null", "promo": "促销文字或 null", "count": "数量或 null", "confidence": 0.0到1.0}],
  "read_text": ["画面里其他可读文字"],
  "uncertain": ["看不清或拿不准的内容"]
}
注意：price 只能来自画面中的价签/标签文字，绝不许根据商品常识猜测价格。"""


def perceive(image_path) -> dict:
    if not image_path or not Path(image_path).exists():
        return _empty("没有提供图片或图片不存在")
    if config.VLM_MODE == "mock":
        return _mock_perceive(image_path)
    return _real_perceive(image_path)


def _empty(note: str) -> dict:
    return {
        "scene": note,
        "items": [],
        "read_text": [],
        "uncertain": [note],
        "evidence": {"source": "mock", "note": "无图片"},
    }


def _mock_perceive(image_path) -> dict:
    """离线标注文件感知：与示例图生成脚本配套，保证可复现。"""
    sidecar = config.PERCEPTIONS_DIR / (Path(image_path).name + ".json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["evidence"] = {
            "source": "mock",
            "image": Path(image_path).name,
            "note": "离线标注文件",
        }
        return data
    return _empty(
        "mock 模式未找到标注文件 %s；请先运行 scripts/make_sample_image.py，"
        "或自行为图片添加 data/perceptions/<图片名>.json" % sidecar.name
    )


def _real_perceive(image_path) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=config.VLM_API_KEY, base_url=config.VLM_BASE_URL)
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    data_url = "data:image/jpeg;base64,%s" % b64

    # 不同厂商对 image_url 的格式支持不一：
    #   - 多数（通义/智谱/OpenAI）要求对象形式 {"url": ...}
    #   - kimi-k3 等要求字符串形式，对象形式会 400 或返回空内容
    #   - 推理型模型可能在 max_tokens 内耗尽思考额度导致 content 为空
    # 策略：按顺序尝试，取第一个"非空可解析 JSON"的结果。
    attempts = [
        {"type": "image_url", "image_url": {"url": data_url}},   # 对象形式
        {"type": "image_url", "image_url": data_url},            # 字符串形式
    ]
    errors = []
    for part in attempts:
        kwargs = dict(
            model=config.VLM_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PERCEPTION_PROMPT}, part]}],
            temperature=0.1,
            max_tokens=1200,
        )
        try:
            resp = _create_compat(client, kwargs)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e)[:160])
            continue
        text = (resp.choices[0].message.content or "").strip()
        data = _extract_json(text)
        if data is not None:
            data["evidence"] = {"source": "vlm", "model": config.VLM_MODEL,
                                "image": Path(image_path).name}
            return data
        errors.append("返回内容非合法 JSON：%s" % text[:120])

    raise RuntimeError(
        "VLM 感知失败（模型 %s）。已尝试对象形式与字符串形式的 image_url，"
        "均未得到合法 JSON。各次尝试的错误/输出：%s | 建议：换用支持 base64 "
        "图片与 JSON 输出的多模态端点（如 qwen-vl-plus / glm-4v），或调大 "
        "max_tokens。" % (config.VLM_MODEL, " || ".join(errors))
    )


def _create_compat(client, kwargs):
    """供应商兼容层：部分模型（如 kimi-k3）不接受 temperature 参数，
    遇到此类 400 错误时自动去掉该参数重试一次。"""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        if "temperature" in str(e) and "temperature" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
            return client.chat.completions.create(**kwargs)
        raise


def _extract_json(text: str):
    """容错地从模型输出里抽取 JSON 对象（截取首个 { 到最后一个 }）。"""
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None
