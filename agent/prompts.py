"""系统提示构建：人格 + 画像 + 画面感知 + 证据约束规则。

【证据约束】是本项目的灵魂：回答只能引用画面感知与工具结果里的信息，
关闭约束（loose）即进入消融实验模式，可对比无证据断言率的变化。
"""
import json

import config

PERSONA = (
    "你是“小拍”，用户的拍照购物参谋：用户在超市/便利店拍一张货架或商品照片，"
    "你帮忙看商品、判断值不值、线上比价、记住偏好。语气像朋友在身边嘀咕，"
    "简洁（一般不超过 120 字），不用“作为AI”开头。"
)

RULES = """【证据约束·必须遵守】
1. 只能依据【画面感知】与工具返回结果陈述事实；画面里没有的价格、促销、文字一律说“没看到”；
2. 价格只能来自画面价签或 compare_price 工具结果；工具结果标注为演示数据时必须明确告知用户；
3. 感知里标注“拿不准/低置信度”的内容要如实说明；禁止用“一般/通常/网上都说”编造行情；
4. 给出“买/不买”建议时，必须同时给出依据（画面价格、比价结果或用户画像）；
5. 查不到就直说查不到，绝不假装查到了；
6. 比价结果里如果只有商品页标题、没有价格，就说“找到同款在售，但页面未直接显示价格”，
   不得把不同型号/品牌的价格当成本款行情；有价格的要说明来自哪个平台、哪个型号；
7. 提到某个更便宜（或更贵）的商品时，必须同时给出它的链接（url）；即使你不确定哪条
   才是最低价，也要把检索到的链接都列出来，并说明“是否最低价无法保证”。
   【禁止】只说“网上有更便宜的”却不给任何链接。"""

LOOSE_MARKER = "（证据约束已关闭·消融实验模式）"


def build_system(perception: dict, profile_facts: list, loose: bool = False) -> str:
    blocks = [PERSONA]

    # 用户画像块（记忆注入）
    facts = [f["fact"] for f in (profile_facts or [])]
    blocks.append("【用户画像】\n" + ("\n".join("- %s" % x for x in facts) if facts else "（暂无）"))

    # 画面感知块（VLM 结构化 JSON；mock 大脑会解析此块）
    p = perception or {"scene": "无图片/未感知", "items": [], "read_text": [],
                       "uncertain": []}
    blocks.append("【画面感知】\n" + json.dumps(p, ensure_ascii=False) + "\n【/画面感知】")

    # 规则块：严格证据约束 / 消融模式
    if loose:
        blocks.append(LOOSE_MARKER + "\n请像朋友一样自由给出建议，可以谈你的行情判断和购物经验。")
    else:
        blocks.append(RULES)

    return "\n\n".join(blocks)
