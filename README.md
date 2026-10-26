# 小拍 · 拍照购物参谋 Agent（SnapAdvisor）

> 一个**多模态购物决策 Agent**：在超市/便利店拍一张货架或商品照片，它帮你读图、
> 判断值不值、线上比价、记住偏好——并且**每一句结论都必须有证据**。
>
> 技术栈：Python · LangGraph 状态图 · OpenAI 兼容 LLM（对话）+ VLM（视觉） ·
> Function Calling 工具层 · SQLite 记忆 · FastAPI 网页版 · 离线 mock 模式 ·
> 评测与消融实验
>
> **本项目零 API Key 即可完整运行**（离线 mock 模式，含工具调用与评测，可复现）。
>
> **阅读导航**：技术读者请从本文档第 1 节往下读；
> **非技术读者（不看代码）请读 [`docs/项目学习路径（非技术版）.md`](docs/项目学习路径（非技术版）.md)**——
> 目标、模块分工、数据流向、运行方式、可验证的日志与报告、循序渐进的熟悉顺序与自检清单。

---

## 1. 项目简介与背景

### 解决什么问题

逛超市时的真实场景：拿起一件商品，想知道——

- 这个价格**值不值**？
- 线上买是不是更便宜？
- 它符合我的预算/偏好吗？

现有工具帮不上忙：购物 App 的扫一扫只认条码、不结合你的偏好；通用 AI 助手
会**一本正经地编造价格**（"网上一般卖 99 元"——它根本没查）。

### 核心理念：诚实 Agent（Evidence-Grounded Agent）

本项目最重要的设计不是"能对话"，而是**不编造**：

```
回答中出现的每一个价格
  ├── 来自画面价签（VLM 从图片里读出来的）        —— 有证据
  └── 来自 compare_price 工具的返回结果           —— 有证据
        └── 两者都没有 → 必须说"没看到价签 / 没查到"，绝不编
```

这套约束贯穿全链路：VLM 感知提示词 → 系统规则 → 工具 evidence 契约 →
回答后的自动校验（`evidence.py`）→ 评测与消融实验。**消融实验证明**：关闭
证据约束后，无证据断言从 0 上升到 4（见第 8 节）。

### Agent 架构总览

```
                         ┌─────────────────────────────────┐
                         │        LangGraph 状态图           │
                         │                                   │
 用户照片 ──► perceive ──►│ agent ⇄ tools（工具循环）         │
             (VLM 读图)   │   │                               │
                         │   ▼                               │
                         │ confirm（HITL：建议先人工确认）    │
                         │   │ regenerate                    │
                         │   ▼                               │
                         │ finalize ──► 证据校验 + 存档      │
                         └─────────────────────────────────┘
      工具：compare_price（比价） / remember_preference（画像）
            / query_profile（查画像） —— 全部带 evidence
      记忆：SQLite（画像事实 + 对话存档）
      感知：VLM 结构化 JSON（价格只来自画面价签）
```

### 与"烂大街 LLM Demo"的区别

| 普通 LLM Demo | 本项目 |
|---|---|
| 一段 prompt 调 API | LangGraph 显式状态图 + 条件边 + 工具循环出口控制 |
| 模型自由发挥（爱编就编） | 证据契约：无依据价格被程序化检出 |
| 没有记忆 | SQLite 画像 + 对话存档，偏好跨会话生效 |
| 没有评测 | 5 用例评测集 + 工具触发率 + 消融实验 |
| 必须有 API Key | 离线 mock 模式全链路可跑、可复现 |
| 只有终端 | CLI + FastAPI 网页版，手机拍照即可用 |

---

## 2. 环境要求与依赖

| 项 | 要求 |
|---|---|
| 操作系统 | Windows / macOS / Linux（示例图脚本会自动选可用字体） |
| Python | **3.10+**（本项目在 3.13.16 验证通过） |
| 硬件 | 无任何要求（不需要 GPU；mock 模式连网络都不需要） |
| 包管理 | 任意（pip / uv / conda 均可；文档以 uv 与 pip 为例） |

依赖清单（`requirements.txt`）：

```
openai>=1.30          # OpenAI 兼容接口客户端（对话 LLM / VLM 共用）
langgraph>=0.2.0      # Agent 状态图编排
python-dotenv>=1.0    # .env 配置加载
fastapi>=0.110        # 网页版服务
uvicorn>=0.29         # ASGI 服务器
python-multipart>=0.0.9  # 表单/文件上传
requests>=2.31        # 比价 web 模式
pillow>=10.0          # 合成示例货架图（仅 scripts 使用）
pytest>=8.0           # 单元测试
```

**API Key（可选）**：全部不填 = 离线 mock 模式。要接真实模型时再填：

- 对话 LLM：DeepSeek（或任何 OpenAI 兼容接口）
- 视觉 VLM：通义 qwen-vl（DashScope 的 OpenAI 兼容端点，有免费额度）
- 比价搜索：Tavily（可选，免费档够用）

---

## 3. 安装与初始化步骤

```powershell
# 0. 进入项目目录
cd D:\code\snap-advisor

# 1. 创建虚拟环境（二选一）
uv venv .venv --python 3.13          # 方式 A：uv（快）
python -m venv .venv                 # 方式 B：标准库

# 2. 安装依赖
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
#  或
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置环境变量（可选——不配置就是离线 mock 模式）
copy .env.example .env
# 用编辑器打开 .env，按注释填入你的 Key

# 4. 生成示例货架图 + 标注文件（首次演示/评测需要）
.venv\Scripts\python.exe scripts\make_sample_image.py
# 输出：data\sample_shelf.jpg 与 data\perceptions\sample_shelf.jpg.json

# 5. 跑单元测试（18 个用例，离线）
.venv\Scripts\python.exe -m pytest tests -q
# 预期：18 passed
```

**示例数据说明**：`data/sample_shelf.jpg` 是脚本**合成**的货架图（不是真实照片），
包含 4 个商品（保温杯/玻璃水杯/不锈钢饭盒/马克杯）与价签、促销标签。与之配套的
`data/perceptions/sample_shelf.jpg.json` 是离线 mock 感知的"标准答案"标注。这套
组合保证：没有相机、没有 API Key，也能演示和评测完整链路。你可以随时用自己拍
的真实照片替换（配真实 VLM Key 使用）。

---

## 4. 运行与启动方式

以下命令均假设当前目录为项目根目录，并使用项目内虚拟环境。

### 4.1 离线演示（零 Key，推荐第一步）

```powershell
.venv\Scripts\python.exe main.py --demo
```

预期输出（节选）：

```
[画面感知] source=mock, items=4
  - 保温杯  ￥49.90  促销:第二件半价
  ...
[工具调用] compare_price
[小拍回答]
我查到的线上报价：
- 京东：不锈钢真空保温杯 500ml ￥45.00（演示数据）
...
[证据校验] 通过：回答中的价格均有依据
```

### 4.2 指定照片与问题（接真实 Key 后的日常用法）

```powershell
.venv\Scripts\python.exe main.py --image C:\photos\shelf.jpg --question "这个保温杯值吗？帮我比价"
```

### 4.3 多轮对话

```powershell
.venv\Scripts\python.exe main.py --chat
# 默认使用示例图；输入 q 退出
# 试：记住我预算不超过50 → 再问：货架上有什么适合我的？
```

### 4.4 网页版（电脑与手机通用，无需部署上线）

```powershell
.venv\Scripts\python.exe main.py --web            # 默认监听 0.0.0.0:8000
.venv\Scripts\python.exe main.py --web --port 8017
.venv\Scripts\python.exe main.py --web --host 127.0.0.1   # 仅本机可访问
```

启动后会打印三个信息：本机地址、**手机地址**（自动探测局域网 IP，如
`http://192.168.x.x:8000`）、以及**二维码图片路径**（`data/web_qr.png`，
双击打开用手机扫码即可，免手输地址）。

**手机怎么访问（不用部署任何东西）**：
1. 手机与电脑连接**同一个 Wi-Fi**；
2. 电脑上启动 `--web`，Windows 防火墙首次弹窗时点"允许访问"（专用网络）；
3. 手机浏览器扫码（或手输 `http://192.168.x.x:8000`）→ 页面里点"拍照识别"
   直接调起后置相机，拍货架即问。

**页面交互（响应式，自动适配电脑/手机）**：
- 宽屏自动双栏、窄屏自动单列——前端媒体查询自动判断设备，无需后端识别；
- 电脑端上传方式任选：**拖拽图片到投放区**、**Ctrl+V 直接粘贴截图**
  （网购商品页截图粘进来就能问）、点击选择文件；
- 手机端显示"拍照识别 / 从相册选"大按钮；
- 提问后回答**逐字流式输出**（可随时点「停止」；网络不稳时自动回退一次性输出，
  结果完全一致）；
- 候选商品链接在正文生成前就以**可点击卡片**给出（全部候选都列出，并标注
  "是否最低价无法保证"）；
- 可展开查看感知 JSON、工具结果、证据校验徽章（通过/违规）、历史存档；
  未传图片时自动回退示例图。

### 4.5 评测与消融实验

```powershell
.venv\Scripts\python.exe evals\run_eval.py     # 评测（强制 mock，可复现）
.venv\Scripts\python.exe evals\ablation.py     # 消融：证据约束 开 vs 关
.venv\Scripts\python.exe main.py --eval        # 等价入口
```

### 4.6 其他实用入口

```powershell
.venv\Scripts\python.exe main.py --test                  # 单元测试
.venv\Scripts\python.exe main.py --index                 # 重建检索索引（记忆 + 知识库）
.venv\Scripts\python.exe main.py --rag-eval              # 检索评测（召回率 / 排序）
.venv\Scripts\python.exe main.py --rag-ablation          # 检索消融（策略 / 切分 / 注入）
.venv\Scripts\python.exe main.py --loose                 # 消融模式（关约束）
.venv\Scripts\python.exe main.py --loose --confirm       # 消融 + HITL 确认演示
```

---

## 5. 核心功能模块说明

| 模块 | 文件 | 说明 |
|---|---|---|
| **配置中心** | `config.py` | 集中读取 .env；未配 Key 自动落入 mock 模式；路径可环境变量重定向（测试隔离用） |
| **状态定义** | `agent/state.py` | LangGraph 全局状态 `AgentState`（消息/感知/画像/工具结果/确认/违规等字段） |
| **状态图** | `agent/graph.py` | 装配 perceive→agent⇄tools→confirm→finalize 的显式图，条件边控制工具循环与 HITL 重生成 |
| **节点实现** | `agent/nodes.py` | 五个节点 + 两个路由函数；纯函数式（输入 state，返回增量更新） |
| **提示词** | `agent/prompts.py` | 人格 + 用户画像块 + 画面感知块 + **证据约束规则**（消融时替换为松弛版） |
| **统一入口** | `agent/runner.py` | CLI/Web/评测共用的 `run_once()`；自动补生成示例图 |
| **对话客户端** | `llm/client.py` | OpenAI 兼容真实调用 + **离线 mock 大脑**（确定性规则，覆盖 function calling 循环） |
| **视觉感知** | `llm/vlm.py` | 照片 → 结构化 JSON（价格只来自价签）；真实 VLM / 离线标注双模式 |
| **工具注册表** | `tools/registry.py` | 四个工具的 function-calling schema 与分发，输出全部带 evidence |
| **比价工具** | `tools/price.py` | mock（演示数据，明确标注）/ web（Tavily→DDG→Bing）双模式；**绝不静默造假** |
| **长期记忆** | `memory/store.py` | SQLite：画像事实（写入/查询）+ 对话存档 |
| **证据契约** | `evidence.py` | 回答中每个价格必须有依据、引用编号必须真实、演示数据必须声明——本项目的灵魂 |
| **检索层（RAG）** | `rag/` | 切分 / 嵌入 / SQLite 向量+全文 / 混合检索 / 重排 / 索引构建；把记忆变成可召回的证据 |
| **轨迹日志** | `logging_utils.py` | JSONL 逐节点事件（data/trace.jsonl），调试与复盘 |
| **流式通道** | `agent/streaming.py` | ContextVar 事件发射器；未绑定时自动退化为一次性输出（保证结果一致） |
| **流式渲染** | `app/static/index.html` | SSE 帧解析 + 16ms 批量渲染 + 停止/重试/超时回退/链接卡片 |
| **CLI** | `main.py` | demo / 单次 / 多轮 / 网页 / 评测 / 消融 / 建索引 / 检索评测 / 检索消融 / 测试 十种入口 |
| **网页版** | `app/server.py` + `app/static/index.html` | FastAPI 六个接口（含 SSE 流式）+ 单文件前端（无构建工具、无 CDN 依赖） |
| **示例数据** | `scripts/make_sample_image.py` | 合成货架图 + ground truth 标注（幂等可重复运行） |
| **评测** | `evals/` | 5 用例数据集 + 24 条检索用例、跑分脚本、消融脚本，报告落 `data/*.json` |
| **测试** | `tests/` | 41 个用例：工具契约 / 证据检出 / 主图循环 / Web 冒烟 / **流式一致性** / **检索与引用 grounding** |

---

## 6. 目录结构解析

```
snap-advisor/
├── main.py                  # CLI 入口（七种模式）
├── config.py                # 配置中心（.env / 环境变量 / 路径）
├── evidence.py              # 证据契约：无依据价格检出
├── logging_utils.py         # JSONL 轨迹
├── requirements.txt         # 依赖清单
├── .env.example             # 环境变量示例（复制为 .env）
├── LICENSE                  # MIT
│
├── agent/                   # LangGraph 编排层
│   ├── state.py             #   AgentState 状态定义
│   ├── graph.py             #   状态图装配（编译单例）
│   ├── nodes.py             #   节点与路由（感知/决策/工具/确认/收尾）
│   ├── prompts.py           #   系统提示（含证据约束规则）
│   ├── streaming.py         #   流式事件通道（ContextVar 发射器）
│   └── runner.py            #   统一运行入口 run_once()
│
├── llm/                     # 模型接入层
│   ├── client.py            #   对话客户端 + 离线 mock 大脑
│   └── vlm.py               #   VLM 结构化感知
│
├── tools/                   # Agent 工具层
│   ├── registry.py          #   schema + 分发（带 evidence）
│   └── price.py             #   比价（mock/web）
│
├── memory/
│   └── store.py             #   SQLite 画像 + 对话存档
│
├── rag/                     # 检索层（RAG）
│   ├── embed.py             #   文本向量（真实语义 / 离线词面哈希双模式）
│   ├── chunk.py             #   切分（带重叠，保留可溯源 id）
│   ├── store.py             #   SQLite 存 chunk + 向量 BLOB + FTS5 全文索引
│   ├── retrieve.py          #   向量 / BM25 / 混合 RRF / 重排
│   └── index.py             #   索引构建：画像事实 / 历史问答 / 本地知识库
│
├── app/                     # 网页版
│   ├── server.py            #   FastAPI 接口（含 /api/ask/stream 的 SSE）
│   └── static/index.html    #   单文件前端（流式渲染 / 拖拽粘贴 / 响应式）
│
├── scripts/
│   └── make_sample_image.py #   合成示例货架图 + 标注
│
├── evals/
│   ├── dataset.json         #   5 用例评测集
│   ├── rag_dataset.json     #   24 条检索用例（含长文档）
│   ├── run_eval.py          #   跑分脚本
│   ├── rag_eval.py          #   检索评测（Recall@5 / Recall@1 / MRR / 端到端）
│   ├── ablation.py          #   证据约束消融脚本
│   └── rag_ablation.py      #   检索消融（策略 / 切分 / 注入方式）
│
├── tests/                   # 41 个单元测试（离线 mock）
│   ├── test_streaming.py    #   流式 == 一次性、delta 拼接、链接直出
│   ├── test_api_stream.py   #   SSE 协议与接口一致性
│   ├── test_rag.py          #   切分 / 嵌入 / 检索 / 引用 grounding
│   └── ...                  #   工具契约 / 证据检出 / 图循环
│
└── data/                    # 运行期数据（大多不进 git）
    ├── sample_shelf.jpg              # 合成示例图（进 git）
    ├── perceptions/sample_shelf.jpg.json  # mock 感知标注（进 git）
    ├── memory.db                     # 日常记忆库（不进 git）
    ├── rag.db                        # 检索库：chunk + 向量 + 全文索引（不进 git）
    ├── kb/                           # 可选本地知识库（放 *.md 即被索引）
    ├── trace.jsonl                   # 节点轨迹（不进 git）
    ├── uploads/                      # 网页上传的照片（不进 git）
    └── eval_report.json / ablation_report.json   # 评测报告（不进 git）
```

---

## 7. 关键概念讲解

### 7.1 LangGraph 状态图（为什么不用 while 循环）

`agent/graph.py` 把流程声明为显式图：节点是纯函数，边（含条件边）由路由函数
决定。好处：

- **循环出口可控**：`route_after_agent` 决定"继续调工具还是进入确认"，
  `MAX_TOOL_ROUNDS` 达到上限后不再下发工具，杜绝死循环；
- **每一步可观测**：节点进出都有 JSONL 轨迹，出问题能定位到具体环节；
- **HITL 是图的一等公民**：`confirm` 节点可以拒绝草稿建议并回到 `agent`
  重新生成（`regenerate` 边），人工介入不是外挂补丁。

### 7.2 Function Calling 工具循环

`agent` 节点把 `tools/registry.py` 里的 schema 下发给对话模型；模型决定调用
`compare_price`（比价）、`remember_preference`（记偏好）、`query_profile`
（查画像）。`tools` 节点执行后把结果作为 `tool` 消息写回，再回到 `agent`，
直到模型给出不再含工具调用的回答。**工具读写分离**：比价是只读查询，
画像写入只接受用户明确表述的事实（`source=user_statement`）。

### 7.3 证据契约（Evidence Contract）——本项目的灵魂

每个工具返回值都带 `evidence`（source / fetched_at / is_demo）。系统规则
要求回答只能引用【画面感知】与【工具结果】。最后 `evidence.py` 用正则抽取
回答中的全部价格，逐一核对是否落在"允许集合"（感知 items + read_text +
工具 results）里——不在就是"无证据断言"。演示（mock）数据被引用却未声明，
同样记违规。这套契约是**可量化、可测试**的诚实。

### 7.4 离线 Mock 大脑（为什么零 Key 也能跑）

`llm/client.py` 里的 mock 是一套**确定性规则**：解析消息历史判断意图
（比价/记偏好/直接问答），产出与 OpenAI 同形的响应（含 tool_calls）。
因此 function calling 循环、证据校验、评测、消融全部离线可跑且**可复现**
——这对开发、CI 和"跑给别人看"都极其重要。注意：mock 的松弛模式会
**故意**编造 ￥99/￥9.9，专门用来演示证据校验如何抓住编造。

### 7.5 消融实验（Ablation）

`evals/ablation.py` 对同一批用例分别以"约束开/关"跑两遍（loose 标记沿
state 传入，替换系统规则并切换 mock 行为）。已验证结果：**约束开 = 0 违规，
约束关 = 4 违规**。这就是"证据约束有效"的量化证据——也是本项目在简历上
最硬的一条 bullet。

### 7.6 多模态感知的结构化契约

VLM 被要求严格输出 JSON（scene/items/read_text/uncertain），并明确
"price 只能来自画面价签，不许用常识猜"。低置信度内容进 `uncertain`，
下游回答必须如实说"拿不准"。感知失败会显式返回空结果而不是瞎猜。

### 7.7 流式输出（SSE）：逐块推送的协议与容错

**触发条件**（满足全部才走流式，否则自动用一次性接口）：
1. 网页端发起 `POST /api/ask/stream`；
2. 后端为该请求绑定事件发射器（`agent/streaming.py`，ContextVar 传递）；
3. 客户端支持 `fetch` 流式读取（`resp.body.getReader()`）。
未绑定发射器时（CLI / 评测 / 测试 / 回退路径）节点自动走一次性调用，
**图结构、提示词、工具、证据校验完全不变**，这正是"流式结果 == 一次性结果"的来源
（`tests/test_streaming.py`、`tests/test_api_stream.py` 有逐字断言）。

**事件协议**（SSE 标准帧：`data: {json}\n\n`，另有注释帧 `: ping` 作心跳）：

| 事件 | 时机 | 载荷 |
|---|---|---|
| `start` | 连接建立 | `{request_id, modes}`——**开始标识** |
| `phase` | 每个阶段开始 | `{stage, label}`，如"正在看这张图…/正在比价…/正在组织回答…" |
| `perception` | 感知完成 | `{data}` 画面感知 JSON |
| `tool` | 每个工具返回 | `{name, output}` 含 evidence |
| `links` | 有候选链接时（工具返回即推，不等正文） | `{items:[{platform,title,price,url}]}` |
| `delta` | 模型生成中 | `{text}` **正文增量** |
| `rollback` | 该轮改为工具调用 | 撤回本轮已吐出的"边想边说"文字 |
| `violations` | 证据校验完成 | `{items}` |
| `done` | 全部完成 | `{answer, perception, tool_results, violations, links, modes}`——**结束标识**，字段与 `/api/ask` 完全一致 |
| `error` | 图执行异常 | `{message, stage}`——**异常结束标识** |

**分块粒度与推送节奏**：
- 真实模型：直接透传模型原生 chunk（实测 DeepSeek 约 1.9 字符/块），服务端**不做人为限速**，保证最小延迟；
- 离线 mock：按 24 字符/块、25ms 间隔切分播放，让零 Key 演示也有一致的观感；
- 前端以 16ms 为节奏批量渲染（把这段时间收到的增量合并后一次性写入 DOM），避免每个字都触发重排；
- 链接在正文之前推送，用户不必等正文生成完就能点开比价。

**心跳与缓冲**：每 10 秒空闲时发一个注释帧 `: ping`；响应头带
`Cache-Control: no-cache, no-transform`、`X-Accel-Buffering: no`、`Connection: keep-alive`，
用于穿透 nginx 等反代的响应缓冲、维持长连接。

**错误与中断处理**（前端 `runAsk()`）：
| 情形 | 处理 |
|---|---|
| 连接失败 / HTTP 错误 / 一个字都没收到 | **自动回退**到一次性接口 `/api/ask`，结果与流式完全相同 |
| 25 秒无任何数据（含心跳） | 判定僵死 → 中止流 → 同样回退一次性接口 |
| 流已开始但中途断开 | 保留已生成的正文与链接，提示"生成中断"，提供「重试」按钮（不重复消耗已完成的部分） |
| 服务端抛出 `error` 事件 | 同上：保留部分内容 + 明确提示 + 重试 |
| 用户点「停止」 | `AbortController` 中止；服务端收到断连后停止入队（不硬杀线程，让模型调用自然收尾） |
| 浏览器不支持流式 | `resp.body.getReader()` 抛错 → 走同一条回退路径 |

### 7.8 低价与候选链接的直出策略

规则（同时落在提示词与代码里，双保险）：
- 提示词规则 7：提到某个更便宜/更贵的商品必须给出它的链接；**禁止**只说"网上有更便宜的"却不给链接；
- 代码兜底：`finalize_node` 确定性地把检索到的链接（去重、带价格）追加到回答末尾的
  【参考链接】块，无论模型有没有主动写出——因此流式与一次性结果一致，CLI/评测同样受益；
- 链接全部列出（哪怕页面没取到价格），并明确标注"是否最低价无法保证"——
  这与项目的诚实原则一致：不假装知道最低价；
- 前端把链接渲染成可点击卡片（`target=_blank`），手机上可直接跳转到商品页。

### 7.9 检索增强（RAG）：把记忆变成可召回的证据

**为什么要做（这不是硬加的功能）**：原来的记忆是"把画像全量塞进提示词"，
用得越久越塞不下——消融表 C 量化了这个必然失效的过程：

| 记忆条数 | 全量注入 token（估） | 只注入检索到的 Top-5 | 8k 窗口是否放得下 |
|---|---|---|---|
| 20 | 353 | 88 | 是 |
| 100 | 1,763 | 88 | 是 |
| 300 | 5,289 | 88 | 是 |
| 1,000 | 17,630 | 88 | **否（必然被截断）** |

所以检索不是为了赶时髦，而是这里**本来就存在、只是原来做得很糙**的需求。

**什么时候该检索（三问测试，全过才做）**：① 答案藏在大量文本里，靠关键词/SQL
查不准？② 检索不到会直接答错或答不出？③ 用户能直观看到价值？
本项目里真正过这三问的是"回忆用户说过什么/问过什么"；**说明书、政策文档问答
这类场景过不了第 ② 问，所以没有硬塞进来。**

**数据从哪来（三种，全部带 source 可溯源）**：

| source | 内容 | 何时入库 |
|---|---|---|
| `memory_fact` | 用户明确说过的偏好/预算 | `remember_preference` 写入时同步索引 |
| `memory_turn` | 过去的问答（问 + 答） | 每轮问答结束后自动增量索引 |
| `kb_file` | `data/kb/*.md` 本地知识库 | `main.py --index` |

**管线**：切分（300 字 / 重叠 60）→ 嵌入 → SQLite 存（chunk 文本 + 向量 BLOB +
FTS5 全文索引）→ 混合检索（RRF）→ 可选重排 → 作为**工具结果**交给模型 →
引用编号 → `evidence.py` 做引用 grounding 校验。

关键取舍（每个都能讲出理由）：
- **不引入向量数据库**：几千条以内 SQLite + 余弦暴力检索是毫秒级，零额外依赖、
  零额外服务；真到十万条再换 ANN 不迟；
- **向量运算用纯 Python**（未引入 numpy）：1536 维 × 千条也就百毫秒量级，
  换来的是零新增依赖、离线可跑；
- **中文全文检索的坑**：SQLite FTS5 的 unicode61 会把连续汉字当一个整词，
  中文几乎检索不到——这里把中文展开成单字 + bigram 再入库（查询同样处理），
  用最朴素的方式让 BM25 在中文上真正生效；
- **混合检索用 RRF 而不是加权求和**：余弦（0~1）和 BM25（负数）量纲不同，
  强行加权对分布敏感；RRF 只用排名，天然可比；
- **重排在真实模式用 LLM 打分，离线用词面覆盖率**（失败自动降级，不影响主流程）。

**离线伪向量的诚实说明**：未配 `EMBED_API_KEY` 时用的是**词面哈希伪向量**
（汉字单字/bigram 哈希到 256 维），只衡量字面重合——"便宜"和"实惠"不会被判
相似。**它不是语义检索**，库里按 `embed_mode` 隔离存储、检索时只比对同模式
向量，绝不拿伪向量的成绩冒充语义检索。配了 Key 才是真语义嵌入。

**引用与"引用传染"**：检索结果带 `chunk_id`，回答必须写成 `【记忆:chunk_id】`，
`evidence.py` 会校验引用编号是否真的来自本轮检索结果（编造编号 = 违规）。
另外历史回答入库前会**先剥掉引用编号**，否则旧引用会随文本被反复复制，
导致引用指向的不是本轮证据——这是实测中真踩到的坑。

**实测数据**（`python evals/rag_eval.py`，24 条用例，离线可复现）：

```
Recall@5 = 100%    Recall@1 = 83%    MRR = 0.903
端到端：4/4 个回忆类问题主动触发 search_kb，引用 grounding 违规 0 处
```

**消融数据**（`python evals/rag_ablation.py`）：

| A 检索策略 | Recall@5 | Recall@1 | MRR |
|---|---|---|---|
| 纯向量 | 100% | 83% | 0.885 |
| 纯关键词 BM25 | 100% | 79% | 0.889 |
| 混合 RRF | 100% | 83% | 0.903 |
| **混合 + 重排** | 100% | **96%** | **0.979** |

| B 切分大小 | 片段数 | Recall@5 | MRR |
|---|---|---|---|
| 200 字 | 30 | 100% | 0.903 |
| 500 字 | 26 | 100% | 0.910 |
| 1000 字 | 26 | 100% | 0.910 |

> B 组的诚实结论：**在这个数据规模下切分几乎不影响效果**（差异 0.007），
> 因为文档都短、且切分不会把句子切断。真正的差异要等知识库里有长文档才会显现
> ——把结论如实写在这里，而不是编一个"切分很关键"的故事。

**命令**：`main.py --index`（重建索引）、`--rag-eval`、`--rag-ablation`；
网页端 `GET /api/kb` 可查看检索库片段数与嵌入模式。

---

## 8. 配置项说明

复制 `.env.example` 为 `.env` 后按需填写。**全部留空 = 离线 mock 模式。**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 空 | 对话 LLM 的 Key（填了即自动 real 模式） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 任意 OpenAI 兼容端点可换（Kimi/通义/本地 vLLM） |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名 |
| `LLM_MODE` | 自动 | 强制指定 `mock` / `real`（默认按 Key 自动判定） |
| `VLM_API_KEY` | 空 | 视觉模型 Key（填了即 real） |
| `VLM_BASE_URL` | 通义 DashScope 兼容端点 | 视觉模型端点 |
| `VLM_MODEL` | `qwen-vl-plus` | 视觉模型名 |
| `VLM_MODE` | 自动 | 强制指定 `mock` / `real` |
| `PRICE_MODE` | `mock` | `mock`=演示数据；`web`=网页检索比价 |
| `TAVILY_API_KEY` | 空 | web 模式可选的搜索 API（更稳定） |
| `REQUEST_TIMEOUT` | `8` | web 检索超时秒数 |
| `SNAP_CONFIRM` | `0` | `1`=购买建议输出前人工确认（HITL，仅 CLI） |
| `SNAP_LOOSE` | `0` | `1`=全局关闭证据约束（消融用；CLI 也可 `--loose`） |
| `MAX_TOOL_ROUNDS` | `3` | 单轮问答最大工具调用轮数 |
| `SNAP_MEMORY_DB` | `data/memory.db` | 记忆库路径（评测/测试用它隔离） |
| `SNAP_RAG_DB` | `data/rag.db` | 检索库路径（评测/测试用它隔离） |
| `SNAP_TRACE_FILE` | `data/trace.jsonl` | 轨迹日志路径 |
| `EMBED_API_KEY` | 空 | 嵌入服务 Key；**留空则用离线词面伪向量**（非语义） |
| `EMBED_BASE_URL` / `EMBED_MODEL` | OpenAI / `text-embedding-3-small` | 嵌入服务地址与模型（可与对话模型不同供应商） |
| `RAG_MODE` | `hybrid` | 检索策略：`vector` / `bm25` / `hybrid` |
| `RAG_RERANK` | `0` | `1`=对候选重排（真实模式多一次 LLM 调用，实测 MRR 0.903→0.979） |
| `RAG_AUTO_INDEX` | `1` | 问答结束后自动增量索引本轮对话 |

**默认配置的语义**：开箱即 mock，是为了"零成本验证全链路"；接真实能力时
只填 Key 不改代码。`PRICE_MODE=web` 在无 Tavily Key 时会自动尝试
DuckDuckGo → Bing 网页检索（国内网络可能不稳定，失败时返回空结果并如实
说明，绝不假装查到）。

---

## 9. 常见问题与排错指南

**Q1：`UnicodeEncodeError: 'gbk' codec can't encode ...`**
Windows 旧版 PowerShell 控制台是 GBK 编码。本项目已内置兜底（无法编码的
字符会替换为 `?`），如果你在其他入口脚本里遇到，可在开头加：
```python
import sys
sys.stdout.reconfigure(errors="replace")
```
或执行 `chcp 65001` 把控制台切到 UTF-8。

**Q2：`python` 命令是 Windows Store 占位符（无输出/闪退）**
用项目内的解释器直接调用：`.venv\Scripts\python.exe ...`；或用 `uv` 管理
（`uv venv` / `uv pip install`）。本项目验证环境即 uv + Python 3.13。

**Q3：mock 模式说"未找到标注文件"**
说明 `data/perceptions/<图片名>.json` 缺失。运行
`.venv\Scripts\python.exe scripts\make_sample_image.py` 生成示例图与标注；
自建图片需要自行配套标注 JSON（结构参考该文件）。

**Q4：网页版上传真实照片，感知结果是空的**
当前是 mock VLM（只认示例标注文件）。在 `.env` 配置 `VLM_API_KEY`（通义
qwen-vl 有免费额度）即可让任意照片走真实视觉理解。

**Q5：比价结果全是"演示数据"**
正常：默认 `PRICE_MODE=mock`。要真实报价：`.env` 里设 `PRICE_MODE=web`，
建议同时申请 `TAVILY_API_KEY`（免费档）提高稳定性；无 Key 时走
DuckDuckGo/Bing 网页抽取，国内网络不稳定时可能返回空结果（会如实说明）。

**Q6：`--confirm` 没有出现确认提问**
两个原因之一：① 草稿回答里没有购买建议类措辞（约束严格的回答本来就倾向
先陈述事实）；② 运行环境不是交互终端（Web 模式自动放行）。想稳定演示
HITL：用 `--loose --confirm`（松弛模式的回答一定带"建议直接买"）。

**Q7：端口被占用**
`main.py --web --port 8017` 换端口；或关闭占用进程。

**Q8：依赖安装慢/失败**
国内加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`；或直接用 uv
（通常更快）。

**Q9：`langgraph` 版本差异报错**
本项目使用稳定的 `StateGraph / START / END / add_conditional_edges` 核心
API，`langgraph>=0.2.0` 均兼容；若你环境里有旧版本，升级即可：
`uv pip install -U langgraph --python .venv\Scripts\python.exe`。

**Q10：合成示例图上中文显示成方块**
系统缺少中文字体。脚本会依次尝试微软雅黑/黑体，找不到时回退默认字体
（标签会变成英文效果）。Windows 一般不会遇到；Linux 可安装
`fonts-noto-cjk`。

**Q11：`Parameter 'temperature'=0.1 is not supported for xxx model`（400）**
部分模型（如 kimi-k3）不接受 `temperature` 参数。本项目已内置供应商兼容
层：`llm/client.py` 与 `llm/vlm.py` 遇到此类 400 错误会自动去掉
temperature 参数重试一次，无需手工处理。若换用其他有类似限制的模型出现
新参数报错，按同样模式在对应调用处加重试即可。

**Q12：`The provided URL does not appear to be valid`（400，发图片时）**
不同厂商对 `image_url` 的格式支持不一：多数要求对象形式
`{"url": "data:image/jpeg;base64,..."}`，而 kimi-k3 只接受**字符串形式**
（且对象形式可能返回空 content）。`llm/vlm.py` 已实现双形式自动回退并
兼容推理型模型的空输出；若你的端点两种形式都不支持（要求公网图片 URL），
建议换用支持 base64 的多模态端点（qwen-vl-plus / glm-4v 等）。

**Q13：证据校验报"无证据价格"，但那个数看起来有道理？**
两种典型情形：① 模型转述搜索结果时改写了数字（原文可能是"200 多元"，
它说成具体值）——这正是契约要拦的；② 模型自己做的推导计算（如"第二件
半价、两件折合单件约 37.4"）——推导值不在证据集合里，严格契约下一律
记违规。已知的改进方向：把契约扩展为"允许显式声明的算术推导"（推导
过程必须可由证据价格复算），这也是一个很好的面试讨论点。

**Q15：比价总说"查不到有效结果"，或搜出一堆股票、新闻之类的无关内容？**
（2026-10 已修复，根因与处理如下）
根因有两条：① VLM 给出的商品名是营销长标题（如"森马小白鞋女鞋秋季新款休闲
运动板鞋"），整句检索时品牌名"森马"会撞上同名 A 股上市公司，返回一堆财经新闻；
② 检索结果没做相关性过滤，任何页面里的数字都被当成"线上报价"，模型看到垃圾
数据只能回"不敢乱说"。现已做四层处理：**核心词提纯**（剔除新款/秋季/厚底等
修饰词）→ **电商站点定向检索**（京东/天猫/淘宝/拼多多/苏宁/什么值得买）→
**财经新闻与无关结果过滤**（宁可返回空也不喂垃圾）→ **有价格的排前面**；
证据校验还支持"十块/十几元"这类中文数字，避免正常转写被误判违规。
仍存在的客观限制：京东/淘宝有反爬且价格由 JS 渲染，回抓商品页常拿不到价格，
因此"找到同款在售但页面没露价"是正常结果，回答会如实说明而不是编造；要稳定
获取真实价格需接有资质的付费数据源（京东联盟 / 淘宝客等开放平台 API）。

**Q14：手机访问不了网页版（电脑能打开）**
按顺序排查：① 手机与电脑是否连的同一个 Wi-Fi（路由器开了"AP 隔离"
的话会互相不可见，需在路由器设置里关闭）；② Windows 防火墙是否放行：
首次启动弹窗点"允许"，或手动放行（管理员 PowerShell）：
`New-NetFirewallRule -DisplayName "snap-advisor" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`；
③ 确认启动日志里"手机访问"显示的是局域网 IP（192.168.x.x）而不是
127.0.0.1；④ 公司/校园网常隔离客户端设备，换手机热点连接电脑试。

**Q16：网页上文字是一整段蹦出来的，没有逐字效果？**
按顺序排查：① 服务端是否真的在流式——浏览器访问
`curl -N -X POST -F "question=xxx" http://127.0.0.1:8000/api/ask/stream`，
应能看到 `data: {"type":"delta"...}` 逐帧输出；若只有一两帧，说明模型或
反代在缓冲（检查响应头 `X-Accel-Buffering: no` 是否生效，或是否经过了
会缓冲响应的反代/内网代理）。② 若为 mock 模式，分块是 24 字符/块 @25ms，
观感接近逐字但节奏固定。③ 若前端报"生成中断"并回退到一次性输出，
说明流式连接被中途切断（见 Q17）。④ 想彻底关掉流式：把
`app/static/index.html` 里 `runAsk()` 的请求地址改为 `/api/ask` 即可，
其余逻辑不受影响（这正是两套路径结果一致的好处）。

**Q17：出现"生成中断"提示，或 25 秒后自动回退？**
这是**设计好的降级链路**，不是故障：流式连接失败/超时/不支持时，
前端会自动改调一次性接口 `/api/ask`，拿到与流式完全相同的结果。
若频繁发生，多半是网络层问题：① 手机与电脑间的 Wi-Fi 不稳定（先换热点
验证）；② 公司/校园网或路由器会切断长连接（可把心跳间隔调小：改
`app/server.py` 的 `HEARTBEAT_INTERVAL`）；③ 反代缓冲了响应（见 Q16）。
若拿到的内容被截断，前端会保留已生成的部分并给出"重试"按钮，
不会让你白等。

**Q19：状态条一直显示"正在组织回答…"，是不是卡死了？**
（2026-10 已修复）大概率不是卡死，而是**状态条根本没被隐藏**：状态条元素的
样式是 `display:flex`，这条作者样式会压过浏览器默认的 `[hidden]{display:none}`，
于是 JS 里 `$('status').hidden = true` 形同无效——答案早就生成完了，那一行却
永远停在最后一个阶段文案上。修复方式是显式声明 `.status[hidden]{display:none}`，
并加了回归测试（`tests/test_api_stream.py::test_status_bar_can_actually_hide`）
与阶段文案清空。**判断是否真的卡住**：看答案正文和链接是否已经出现——出现了
就说明流程已结束（以 `data/trace.jsonl` 里的 `finalize` 记录为准）。
另外现在有三重兜底，真卡住也会自己报错：① 服务端硬超时 180 秒必发 `error`；
② 前端 45 秒收不到任何字节判定断连；③ 只有心跳、120 秒无业务进展也会中止并
给出「重试」——心跳只证明连接活着，不再掩盖"服务端无进展"。

**Q20：问"我上次说过什么"，回答却说没检索到？**
按顺序排查：① `GET /api/kb` 看 `chunks` 是否为 0——是 0 说明检索库没建，
跑一次 `main.py --index`，或提一个回忆类问题（首次会自动建索引）；
② 记忆里本来就没内容：先说一句"记住：我不买含硅油的洗发水"，再问一次；
③ 未配 `EMBED_API_KEY` 时是**词面检索**，同义词（"便宜" vs "实惠"）召回不到，
这是预期行为，配 Key 后才是语义检索；④ 想看检索器本身的成绩，跑
`evals\rag_eval.py`，会给出 Recall@5 / Recall@1 / MRR 与端到端触发率。

**Q18：回答里没有候选商品链接？**
链接来自比价工具结果里的 `url` 字段：① 若为 `PRICE_MODE=mock`（离线演示），
结果是演示数据、本就没有链接，属正常；② 若为 web 模式但仍无链接，
说明检索全部失败或被相关性过滤丢弃（见 Q15），此时回答会如实说"没查到"，
不会凭空给链接——这是诚实原则；③ 想总是拿到链接，确保
`TAVILY_API_KEY` 已配置且网络可访问（国内网络下备用源 DDG/Bing 可能不通）。

---

## 10. 上手练习建议

按难度递进，每一步都只改一两个文件，是理解本项目的最短路径：

1. **改人格与语气**：编辑 `agent/prompts.py` 的 `PERSONA`（比如更毒舌），
   跑 `--demo` 对比回答风格 → 理解系统提示的作用边界。
2. **加一个新工具**：在 `tools/registry.py` 仿照 `compare_price` 加一个
   `search_reviews`（搜评价摘要，可以先 mock），schema + 分发两处各几行 →
   跑 demo 看模型是否会调用它（问"这牌子口碑怎么样"）。
3. **给评测集加用例**：在 `evals/dataset.json` 加一条（比如"马克杯线上
   多少钱"→ 期望 `compare_price`），跑 `evals\run_eval.py` 看通过与否 →
   学会用评测驱动改动。
4. **玩转检索**：先说一句"记住：我不买含硅油的洗发水"，再问"我上次说过买洗发水
   要注意什么"，观察它是否触发 `search_kb`、回答是否带 `【记忆:...】` 编号；
   然后 `main.py --rag-eval` 看 Recall@5，`--rag-ablation` 看"重排到底值不值"
   那张表——**用数据说话而不是凭感觉**，是这个项目最值钱的习惯。
6. **接真实比价**：申请 Tavily Key 填入 `.env`，`PRICE_MODE=web`，对比
   演示数据与真实网页抽取的差异，观察 evidence 里 source/engine 的变化。
7. **接真实 VLM 与 LLM**：填 `VLM_API_KEY` 与 `DEEPSEEK_API_KEY`，用手机
   拍真实货架照片走 `--image` 流程，观察真实模型是否遵守证据约束、违规
   检出是否工作（这一步最能体会"诚实 Agent"的价值）。
8. **做真实消融**：真实 LLM 下分别跑严格/松弛模式各 10 个问题，统计
   无证据断言率——你会得到属于自己的、可写进简历的消融数据。
9. **玩转流式**：`curl -N -X POST -F "question=保温杯值不值？帮我比价"
   http://127.0.0.1:8000/api/ask/stream` 直接看 SSE 原始帧，认全
   start/phase/delta/links/done；再在前端点一次「停止」，观察中断时
   内容如何保留、如何重试——这是"工程化程度"的直观体现。
10. **进阶（练架构）**：给 `agent/graph.py` 加 SQLite checkpointer
   （`langgraph` 的 checkpointer 生态），让中断的对话可以从上次的
   finalize 状态恢复——这会把"断点续作"能力补全。

---

## 附：一句话总结

> 拍照 → VLM 只报画面里看得见的 → 工具查到的都带证据 → 回答里的每个价格
> 都被程序核对过 → 查不到就直说查不到。**别人生成内容，这个项目验证内容。**

MIT License · 独立设计与开发 · 反馈与改进欢迎开 Issue
