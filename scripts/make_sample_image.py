"""生成合成示例货架图 + VLM mock 标注文件。

为什么用合成图：让项目"零外部素材"即可演示与评测——
  - data/sample_shelf.jpg              合成货架图（真实 VLM 也能读出文字）
  - data/perceptions/sample_shelf.jpg.json   离线 mock 感知的 ground truth

运行：python scripts/make_sample_image.py（幂等，可重复运行）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import config  # noqa: E402

# 商品三元组：(名称, 价签价格, 促销)
ITEMS = [
    ("保温杯", "49.90", "第二件半价"),
    ("玻璃水杯", "12.50", None),
    ("不锈钢饭盒", "35.00", "满2件减5"),
    ("马克杯", "19.90", None),
]

_GROUND_TRUTH = {
    "scene": "超市货架（合成演示图），四个商品",
    "items": [
        {"name": n, "price": p, "promo": promo, "count": "1", "confidence": 0.95}
        for n, p, promo in ITEMS
    ],
    "read_text": ["示例货架（合成演示图）"],
    "uncertain": [],
}


def _font(size: int, bold: bool = False):
    """优先中文字体（Windows 微软雅黑/黑体），失败则回退默认字体。"""
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_cell(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, item):
    name, price, promo = item
    # 商品底色块
    d.rectangle([x, y, x + w, y + h], fill="#ffffff", outline="#d8cfc0", width=2)
    d.rectangle([x + 16, y + 18, x + w - 16, y + h - 56], fill="#cfe3f5",
                outline="#8fb8d8", width=1)
    d.text((x + 24, y + h // 2 - 46), name, font=_font(26, True), fill="#23415e")
    # 价签（黄底红字，模拟真实价签观感）
    d.rectangle([x + 16, y + h - 50, x + w - 16, y + h - 14], fill="#fff3bf",
                outline="#e0b400", width=1)
    d.text((x + 28, y + h - 44), "¥" + price, font=_font(24, True), fill="#c0392b")
    if promo:
        d.text((x + 16, y + h + 8), "促销:" + promo, font=_font(18), fill="#7c6a30")


def main():
    config.ensure_dirs()
    W, H = 900, 660
    img = Image.new("RGB", (W, H), "#f6f2e9")
    d = ImageDraw.Draw(img)

    d.text((28, 18), "示例货架（合成演示图）", font=_font(30, True), fill="#374151")
    d.line([(0, 78), (W, 78)], fill="#c9bfa8", width=3)

    # 两行货架，每行两个商品
    cells = [(36, 100, 396, 220), (468, 100, 396, 220),
             (36, 390, 396, 220), (468, 390, 396, 220)]
    for cell, item in zip(cells, ITEMS):
        _draw_cell(d, *cell, item)

    # 货架隔板
    d.rectangle([0, 352, W, 366], fill="#c9bfa8")
    d.rectangle([0, 642, W, 656], fill="#c9bfa8")

    img.save(str(config.SAMPLE_IMAGE), quality=92)
    print("已生成示例图: %s" % config.SAMPLE_IMAGE)

    # 写入 VLM mock 感知的 ground truth 标注
    sidecar = config.PERCEPTIONS_DIR / (config.SAMPLE_IMAGE.name + ".json")
    sidecar.write_text(json.dumps(_GROUND_TRUTH, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print("已生成标注文件: %s" % sidecar)


if __name__ == "__main__":
    main()
