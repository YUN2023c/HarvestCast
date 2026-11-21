# Third-Party Notices · 第三方组件与数据源声明

本项目（HarvestCast）以 **MIT** 协议发布。以下逐项声明第三方代码、数据与服务的来源与授权。

**协议策略**：只允许 MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause / PSF / ISC。
**强传染协议（GPL-2.0、GPL-3.0、LGPL-2.1、LGPL-3.0、AGPL-3.0）一律不引入**，包括间接依赖。

机器可读版本见 [`third_party_licenses.json`](./third_party_licenses.json)（`/api/settings` 与「系统设置」页读取此文件）。

---

## 1. 代码依赖

### 1.1 运行时依赖

**无。** 系统全部功能（数据清洗、OLS、GBRT、SARIMAX-CLS、回测、HTTP 服务、前端图表）均基于 Python 3.9+ 标准库实现，`pip install` 不是必需步骤。

### 1.2 可选依赖（缺失时自动回退，不影响功能）

| 组件 | 版本建议 | 协议 | 用途 | 是否修改 |
|---|---|---|---|---|
| [statsmodels](https://www.statsmodels.org/) | ≥0.14 | **BSD-3-Clause** | 完整 SARIMAX 对照实现 | 否，仅 import |
| [LightGBM](https://github.com/microsoft/LightGBM) | ≥4.0 | **MIT** | GBM 后端加速 | 否，仅 import |
| [NumPy](https://numpy.org/) | ≥1.24 | **BSD-3-Clause** | 上述两库的间接依赖 | 否 |
| [pandas](https://pandas.pydata.org/) | ≥2.0 | **BSD-3-Clause** | AKShare / statsmodels 的间接依赖 | 否 |
| [AKShare](https://github.com/akfamily/akshare) | ≥1.12 | **MIT** | 行情抓取适配器 | 否，仅调用 `futures_zh_daily_sina` |

调用位置：

- `src/models/sarimax.py::fit_with_statsmodels` — statsmodels
- `src/models/gbm.py`（`LGB_AVAILABLE` 分支）— LightGBM
- `src/data/ingest.py::fetch_market_akshare` — AKShare

### 1.3 参考但未引入

| 组件 | 协议 | 未引入原因 |
|---|---|---|
| [Prophet](https://github.com/facebook/prophet) | MIT | 季节项已由内置基准覆盖，引入只增加依赖 |
| [Qlib](https://github.com/microsoft/qlib) | MIT | 研究平台级依赖过重，与本项目"零重依赖"目标冲突 |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | MIT | 本项目不对 LLM 做微调 |
| [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) / [TradingAgents](https://github.com/TauricResearch/TradingAgents) | Apache-2.0 | 多智能体"由 LLM 给结论"与本项目"LLM 不预测价格"的核心约束冲突 |

### 1.4 明确排除（强传染协议）

| 组件 | 协议 | 说明 |
|---|---|---|
| Backtrader | GPL-3.0 | 回测逻辑自研（约 200 行），无需引入 |
| PyQt / PySide | GPL / LGPL | 前端为原生 HTML，无桌面 GUI 依赖 |
| TA-Lib | BSD（但部分发行版打包受限） | 未使用技术指标 |

---

## 2. 数据来源

> **重要：代码协议 ≠ 数据授权。** 上述 MIT/BSD 库抓取或引用的数据，其版权与商用许可由原始数据提供方决定。

| 数据 | 来源 | 授权状态 | 本项目处理方式 |
|---|---|---|---|
| 期货行情（默认通道） | 新浪财经期货日线接口（`stock2.finance.sina.com.cn`） | 无正式商用授权，仅限个人研究与演示 | 标准库 urllib 直接请求，不含第三方代码；不随附抓取结果 |
| 期货行情（备用通道） | AKShare → 新浪财经 / 交易所公开行情 | 代码 MIT；数据需逐源核验 | 仅提供适配器代码 |
| 基本面因子 | LLM 检索的公开报告、公告、新闻 | 版权归原发布方 | 每条记录保存 `source_url` + 原文摘录，供人工复核；不做批量转载 |
| 演示数据 | `src/data/ingest.generate_demo()` 合成 | 本项目生成，MIT | **仅用于流程演示与单测，禁止用于真实决策** |

数据风险与商用检查清单见 [`docs/DATA_SOURCES.md`](./docs/DATA_SOURCES.md)。

---

## 3. 字体与视觉资源

- 字体：仅使用系统字体栈（`-apple-system` / `SF Pro Text` / `PingFang SC` / `Microsoft YaHei`），不内嵌任何字体文件，无授权问题；
- 图标：全部为手写内联 SVG 路径（24×24 网格，`stroke` 描边），不使用第三方图标库；
- 图表：手写内联 SVG，不使用 ECharts / Chart.js / D3 等库；
- 图片：无。避免任何形式的素材版权风险。

---

## 4. 设计体系来源说明

UI 采用「iOS 26 Liquid Glass + 低饱和墨绿」设计语言的工程化实现（玻璃层次、hairline 描边、弹簧曲线、降级策略），
为**风格借鉴**而非复制任何 Apple 私有资源；所有 token、组件与代码均为本项目原创实现。

---

## 5. 合规自查

- [x] 无 GPL / LGPL / AGPL 依赖（含间接依赖）
- [x] 所有第三方调用点在代码注释中标明来源与协议（见下表）
- [x] 不随附任何真实行情数据
- [x] 不内嵌第三方字体、图标、图片
- [x] 对外文档包含免责声明与数据源风险提示

调用点注释对照：

| 文件 | 注释内容 |
|---|---|
| `src/models/sarimax.py` | statsmodels（BSD-3-Clause）可选路径说明 |
| `src/models/gbm.py` | LightGBM（MIT）可选后端说明 |
| `src/data/ingest.py` | AKShare（MIT）及其上游数据授权提示 |
| `docs/RESEARCH.md` §11 | 全部参考项目的仓库链接与协议核验结果 |

---

## 6. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-21 | 初版：确认运行时零依赖，登记 5 个可选依赖与 3 类数据源 |

如发现遗漏或协议标注有误，请提交 issue，我们会在 72 小时内核实并更正。
