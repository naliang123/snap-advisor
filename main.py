"""小拍 · 拍照购物参谋 —— 命令行入口。

常用命令（详细说明见 README.md）：
  python main.py --demo                          # 零 Key 离线演示（默认行为）
  python main.py --image 图片.jpg --question "保温杯值吗？帮我比价"
  python main.py --chat                          # 多轮对话（默认用示例图）
  python main.py --web                           # 网页版 http://127.0.0.1:8000
  python main.py --loose --confirm               # 消融模式 + HITL 确认演示
  python main.py --eval                          # 离线评测
  python main.py --ablation                      # 消融实验（证据约束开/关对比）
  python main.py --test                          # 单元测试
"""
import argparse
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser(description="小拍 · 拍照购物参谋 Agent")
    ap.add_argument("--demo", action="store_true", help="零 Key 离线演示（默认）")
    ap.add_argument("--image", help="照片路径（jpg/png）")
    ap.add_argument("--question", "-q", help="你的问题")
    ap.add_argument("--chat", action="store_true", help="进入多轮对话（终端）")
    ap.add_argument("--web", action="store_true", help="启动网页版 (FastAPI)")
    ap.add_argument("--port", type=int, default=8000, help="网页版端口，默认 8000")
    ap.add_argument("--host", default=None,
                    help="监听地址，默认 0.0.0.0（允许手机在同一 Wi-Fi 下访问）；"
                         "仅本机可访问请用 --host 127.0.0.1")
    ap.add_argument("--loose", action="store_true",
                    help="关闭证据约束（消融实验模式，会出现无证据断言）")
    ap.add_argument("--confirm", action="store_true",
                    help="购买建议输出前人工确认（HITL 演示）")
    ap.add_argument("--eval", action="store_true", help="运行离线评测")
    ap.add_argument("--ablation", action="store_true", help="运行消融实验")
    ap.add_argument("--index", action="store_true",
                    help="重建检索索引（记忆 + 本地知识库）")
    ap.add_argument("--rag-eval", action="store_true", help="运行检索评测（召回率/排序）")
    ap.add_argument("--rag-ablation", action="store_true", help="运行检索消融实验")
    ap.add_argument("--test", action="store_true", help="运行单元测试")
    return ap.parse_args()


def _banner(config):
    m = config.mode_summary()
    print("=" * 62)
    print("小拍 · 拍照购物参谋 Agent（LangGraph 多模态购物决策 Agent）")
    print("LLM: %s(%s)  VLM: %s(%s)  比价: %s  证据约束: %s" % (
        m["llm_mode"], m["llm_model"], m["vlm_mode"], m["vlm_model"],
        m["price_mode"], "开" if m["evidence_strict"] else "关(消融)"))
    print("=" * 62)


def _print_result(result):
    print("\n" + "-" * 62)
    p = result.get("perception") or {}
    ev = p.get("evidence") or {}
    items = p.get("items") or []
    print("[画面感知] source=%s, items=%d" % (ev.get("source", "?"), len(items)))
    for it in items:
        price = ("￥" + str(it.get("price"))) if it.get("price") else "无价签"
        promo = ("  促销:" + str(it["promo"])) if it.get("promo") else ""
        print("  - %s  %s%s" % (it.get("name"), price, promo))

    calls = [t["tool"] for t in result.get("tool_results", [])]
    print("[工具调用] %s" % (", ".join(calls) if calls else "（无）"))

    print("[小拍回答]")
    print(result.get("final_answer", ""))

    viol = result.get("violations") or []
    if viol:
        print("[证据校验] 发现 %d 处违规：%s" % (len(viol), viol))
    else:
        print("[证据校验] 通过：回答中的价格均有依据")
    print("-" * 62)


def _run_single(question, image, loose):
    from agent import runner
    import config
    runner.ensure_sample()
    result = runner.run_once(
        question, image or str(config.SAMPLE_IMAGE), loose=loose
    )
    _print_result(result)
    return result


def _chat(args):
    from agent import runner
    import config
    runner.ensure_sample()
    image = args.image or str(config.SAMPLE_IMAGE)
    print("进入多轮对话（输入 q 退出；图片：%s）" % image)
    messages = None
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break
        result = runner.run_once(q, image, loose=args.loose, messages=messages)
        _print_result(result)
        # 携带本轮消息历史进入下一轮
        messages = [m for m in result.get("messages", [])
                    if m.get("role") != "system"]
    print("再见。")


def main():
    global config
    args = parse_args()
    # Windows GBK 控制台兜底：无法编码的字符替换为 ?，避免崩溃
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    # SNAP_CONFIRM 必须在 import config 之前写入环境变量
    if args.confirm:
        os.environ["SNAP_CONFIRM"] = "1"

    import config  # noqa: E402  (晚于环境变量设置，故需要)

    if args.test:
        sys.exit(os.system('"%s" -m pytest tests -q' % sys.executable) and 1)
    if args.eval:
        _run_script("evals/run_eval.py")
        return
    if args.ablation:
        _run_script("evals/ablation.py")
        return
    if args.rag_eval:
        _run_script("evals/rag_eval.py")
        return
    if args.rag_ablation:
        _run_script("evals/rag_ablation.py")
        return
    if args.index:
        from rag import index as rag_index
        stats = rag_index.reindex_all()
        print("检索索引已重建：%s" % stats)
        return
    if args.web:
        import socket
        import uvicorn

        host = args.host or "0.0.0.0"
        port = args.port

        # 探测局域网 IP（UDP connect 不真正发包，只查路由表）
        ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            try:
                ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                pass

        print("网页版已就绪：")
        print("  本机访问   http://127.0.0.1:%d" % port)
        if host == "0.0.0.0" and ip != "127.0.0.1":
            lan_url = "http://%s:%d" % (ip, port)
            print("  手机访问   %s   （需与电脑连同一 Wi-Fi）" % lan_url)
            qr_path = _make_qr(lan_url)
            if qr_path:
                print("  二维码     %s （双击打开图片，用手机扫码）" % qr_path)
            else:
                print("  （未安装 qrcode 库，无法生成二维码图片：uv pip install qrcode）")
            print('  提示：Windows 防火墙首次弹窗时请选择"允许访问"（专用网络）；'
                  "若手机打不开，见 README 常见问题 Q14。")
        uvicorn.run("app.server:app", host=host, port=port)
        return

    _banner(config)
    if args.chat:
        _chat(args)
        return
    if args.image or args.question:
        if not args.question:
            print("使用 --image 时请同时提供 --question")
            sys.exit(1)
        _run_single(args.question, args.image, args.loose)
        return

    # 默认：离线 demo
    _run_single(
        "货架上的保温杯值不值得买？顺便帮我看看有没有更便宜的。",
        None, args.loose,
    )


def _make_qr(url):
    """生成局域网地址二维码 PNG（可选依赖 qrcode；失败打印原因并返回 None）。"""
    try:
        import qrcode
        import config as _cfg
        _cfg.ensure_dirs()
        path = _cfg.DATA_DIR / "web_qr.png"
        qrcode.make(url).save(str(path))
        return path
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _run_script(rel_path):
    """以模块方式执行仓库内脚本（evals 等非包目录）。

    脚本以模块方式加载时 __name__ 不是 __main__，因此这里显式调用其 main()。
    """
    import importlib.util
    path = config.ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "main", None)
    if fn:
        fn()


if __name__ == "__main__":
    main()
