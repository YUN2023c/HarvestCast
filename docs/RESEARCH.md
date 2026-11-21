# 调研报告：10 个方向 + 开源项目与协议核验

> 本文件是设计前置产物。所有架构选择都可追溯到下面的某条结论；反过来，凡是本文件没有支撑的设计，都不应进入代码。
> 调研时间：2026-11。核验方式：官方文档 / 论文 / 仓库 LICENSE 元数据。

## 0. 结论先行

| # | 方向 | 对本项目的直接决定 |
|---|---|---|
| 1 | 农产品中长期预测建模思路 | 以**供需基本面 + 季节性 + 均值回归**为主干，期货价作为前瞻锚，不做高频 |
| 2 | 远月 vs 现货季节性 | 目标必须是**收获窗口**；远月价入模作基差特征，并作为校验锚 |
| 3 | LLM 结构化抽取工程 | 三层校验（结构/约束/业务）+ 逐字段引用 + 置信度门控 |
| 4 | SARIMAX/Prophet 落地 | 月度 s=12、BIC 选阶、残差白噪声检验、区间随跨度扩张 |
| 5 | LightGBM 长周期 | **直接多步**、季节性滞后、留出验证定区间、承认无法外推 |
| 6 | Schema 标准化 | JSON Schema 2020-12 + strict 三约束 + 枚举收敛 |
| 7 | 可商用轻量架构 | 四层、文件即接口、零重依赖、协议白名单 |
| 8 | 无数据库数据管线 | CSV + 原子写 + `.bak` + 审计日志 + 可 git diff |
| 9 | API/手动双模式 | 同一模板 + 同一 Schema + 同一校验器 + 模板指纹 |
| 10 | 供需因子与时序融合 | 时序做基准、因子做修正、弹性做情景，三者职责分离 |

---

## 1. 农产品中长期价格预测的通用建模思路

**业界常见四类方法**（FAO / USDA / 世界银行口径）：基本面分析、技术分析、计量建模、情景与敏感性分析，外加"期货市场作为前瞻信号"。

关键共识：
- 中长期（6–12 个月）预测的可信来源是**供给（面积/单产/产量/库存/进口）+ 需求（饲用/加工/出口）+ 政策**，而不是价格形态；
- 学术与工业界都强调农产品价格的三要素：**强季节性、供需滞后、均值回归**。Fraunhofer ITWM 的两因子模型明确把"seasonality / long-term trend / short-term fluctuation with mean reversion"作为建模目标，再用外部因子（降水、气温、消费、产量）驱动情景；
- 单一模型容易过拟合，组合模型（统计 + ML + 分解重构）在文献中普遍优于单模型，但**必须先有朴素基准**。

→ 落到本项目：
- 目标定为"收获窗口均价"，不是某日收盘；
- 主干 = 季节基准 + SARIMAX + GBM，三者互补；
- 因子表只收基本面事实，不收舆情情绪（情绪难以稳定量化，且易被 LLM 编造）。

参考：
- Fraunhofer ITWM, *Price modelling of agricultural commodities under consideration of external factors* — https://www.itwm.fraunhofer.de/en/departments/fm/latest-news/current-research-topics/price-modelling-of-agricultural-commodities.html
- Global Trade Connect, *Global Commodity Price Forecasting Methods*（方法分类与局限）— https://global-trade-connect.com/global-commodity-price-forecasting-methods
- MDPI Electronics 13(22):4362, *AI Tools for the Agriculture Value Chain*（数据需求与混合模型）— https://mdpi.com/2079-9292/13/22/4362

---

## 2. 期货远月价 vs 现货季节性

核心概念是**基差**：`基差 = 现货价 − 期货价`，由持仓成本（仓储、保险、运输、资金）与**季节预期**共同决定。

- 收获季现货充裕 → 现货承压，期货相对现货升水（负基差），这是农产品特有的**季节性 backwardation**；不能读成"市场看空"；
- 青黄不接期现货紧张 → 现货升水（正基差）；
- 套保者的真实售价 ≈ 期货价 + 预期基差，因此**只用期货价做预测会系统性偏离**；
- USDA ERS 的季节均价预测方法就是"期货价 + 历史基差 + 销售进度"的组合。

→ 落到本项目：
- 预测标的 = 收获窗口均价（对应真实销售时点）；
- `basis = log(far_close) − log(close)` 作为特征；
- 校验层增加"预测价 vs 远月价偏离 ≤18%"，偏离过大必须提示（可能是模型错，也可能是跨期机会）。

参考：
- *Agricultural futures basis*（持仓成本与季节基差）— https://financial-wiki.vercel.app/agricultural-futures-basis
- *Agricultural Price Forecasting and Basis Risk*（`expected local cash = futures + expected basis`）— https://theagriculturedata.com/agricultural-price-forecasting-basis-risk
- UW Extension, *Understanding Basis in Grain Marketing* — https://farms.extension.wisc.edu/articles/understanding-basis-in-grain-marketing
- NISM, *Seasonality effect*（播种季升水/收获季贴水的形成机制）— https://nismprofessor.com/glossary/seasonality-effect

---

## 3. LLM 结构化金融数据抽取的工程方案

行业成熟做法已从"给模型一个 PDF 让它吐 JSON"演进为**分阶段管线 + 引用锚定 + 确定性校验**：

| 阶段 | 作用 | 关键控制 |
|---|---|---|
| 分类 | 判断文档类型，选择 schema | 误分类会级联污染后续所有字段 |
| 证据映射 | 为每个待抽字段定位原文位置 | 抽取只能看被映射的证据，幻觉被结构性限制 |
| 抽取 | 按 schema 输出，逐字段给置信度 | 结构化输出 + 字段级 provenance |
| 校验 | 跨字段一致性、数值可复算 | 失败只标记字段，不整篇拒绝 |
| 打分 | 文档级/字段级置信度聚合 | 低于阈值转人工 |

另一条共识：**schema 不能保证数值真实**。约束只能消灭格式漂移；数值对不对，必须靠确定性校验（重算、比对主数据、范围检查）。

→ 落到本项目：
- `prompts/schema.json` 强约束字段与枚举；`evidence_quote` 强制原文摘录；
- 三层校验：结构 → 约束 → 业务（日期越界 / 价格预测表述 / 品种匹配 / 重复项）；
- 置信度 <0.55 或无有效 URL 的记录**不进训练集**，但在 UI 里展示剔除原因。

参考：
- MDPI *Future Internet* 18(6):284, *An Auditable LLM-RAG Architecture for Financial Document Intelligence*（citation anchoring + JSON-schema validation + groundedness）— https://www.mdpi.com/1999-5903/18/6/284/htm
- Cognilium, *The 8-Stage Document Intelligence Pipeline* — https://cognilium.ai/blogs/8-stage-docint-pipeline
- Nixi Consulting, *Trustworthy AI in Finance Data Pipelines*（"结构 ≠ 真实"）— https://www.nixiconsulting.com/blog/trustworthy-ai-finance-data-pipelines

---

## 4. SARIMAX / Prophet 在农产品上的落地范式

- 月度数据取 `s=12`；先看原始序列与滚动均值/方差，再决定 `d/D`，**避免过度差分**（会放大区间）；
- 阶数选择：ACF/PACF 给初值 + AIC/BIC 比较 + 留出期误差；
- 残差必须做 Ljung-Box，p 小说明还有结构未提取；
- 区间随跨度自然变宽，若 95% 区间只覆盖 70% 实际值，说明过度自信；
- 外生变量必须"预测期可得"，否则等于作弊；
- Prophet 的优势是可解释的趋势/多重季节/变点，劣势是加法结构与对复杂交互的建模能力；它适合做**基准与解释**，不适合做唯一模型。

→ 落到本项目：
- SARIMAX 只保留 AR + 季节 AR + 滞后外生（条件最小二乘），BIC 选阶；
- 输出 `order / BIC / σ / R² / Ljung-Box p / DW` 供人核查；
- 保留 `fit_with_statsmodels()` 作为可选对照路径（BSD-3，可商用）；
- 不用 Prophet：其季节项已被季节基准覆盖，引入会增加依赖而不增加信息。

参考：
- statsmodels 官方 ARIMA/SARIMAX 文档与示例 — https://topolearnapp.com/learn/statsmodels/arima
- *SARIMA in Practice*（差分纪律、区间校准、rolling-origin）— https://thelinuxcode.com/sarima-in-practice-seasonal-forecasting-you-can-defend-and-operate-in-2026
- *Time Series Analysis and Forecasting: A Practical Python Guide* — https://rottenwifi.com/time-series-analysis-and-forecasting-a-practical-python-guide-2026-update

---

## 5. 轻量机器学习因子预测框架（LightGBM 长周期）

- LightGBM 本身不是时序模型，靠**特征工程**把序列转成监督学习：滞后、滚动统计、日历特征、外生变量；
- **滞后阶必须 ≥ 预测跨度**，否则会把目标信息漏进特征；
- 多步策略：递归（误差累积、长跨度易发散）vs 直接（每个跨度单独建模、无累积误差）。长跨度推荐直接法；
- 树模型无法外推：预测值必然落在训练标签范围内。极端情景必须换方法；
- 小样本下深度要浅（3–4 层）、叶子样本要够（≥8），并做时间序留出验证；
- 早期停止很关键：多训 200 轮，留出集误差可能反而恶化 20%+。

→ 落到本项目：
- 直接多步，每跨度独立模型；
- 深度 3 / 220 树 / lr 0.05 / 行、列采样 0.85 / 种子固定；
- 区间用"前 75% 训练、后 25% 留出"的残差标准差，而非训练残差；
- 情景改用结构弹性（见 §10），因为树外推不可靠。

参考：
- LightGBM 官方文档 — https://lightgbm.readthedocs.io/
- ACM, *LightGBM-Based Time Series Forecasting for CSI 300*（特征构造与递归预测细节）— https://acm-prod.literatumonline.com/doi/10.1145/3785706.3785755
- *Gradient Boosting for Time Series: Using LightGBM to Forecast*（早期停止的实证）— https://pythonanddatascience.com/blog/gradient-boosting-for-time-series-using-lightgbm-to-forecast
- *Lag Features*（horizon-lag 规则与 CCF 选滞后）— https://medium.com/data-science-collective/lag-features-teaching-your-model-to-remember-the-past-005a0493ed32

---

## 6. 金融 JSON 结构化抽取的 Schema 标准化

- 事实标准是 **JSON Schema Draft 2020-12**：OpenAI 结构化输出、Anthropic tool use、MCP 都接受它；
- strict 模式三条硬约束（违反即 400）：
  1. 每个 object 必须显式 `additionalProperties: false`；
  2. 所有属性必须列入 `required`；"可选"要用 `type: ["T","null"]` 表达，而不是从 required 删掉；
  3. 根必须是 object，顶层不能是 `anyOf`；
- `enum` 是最有效的收敛手段；`description` 是给模型的第二份提示词；
- 需要处理 refusal 与截断（`finish_reason=length`），不能当成普通解析失败。

→ 落到本项目：
- `schema.json` 逐层 `additionalProperties:false`，全字段 required，可空字段用 `[T,null]`；
- `factor_id`、`direction`、`impact`、`magnitude`、`mode` 全部枚举；
- 客户端显式检测截断并提示；refusal 作为一等结果处理。

参考：
- *JSON Schema for LLM structured output*（2020-12 与五种关键关键字）— https://712tools.com/blog/json-schema-for-llm-structured-output
- *Strict JSON Schema for OpenAI Structured Outputs*（三条规则的机械修正）— https://www.aijsons.com/blog/json-schema-openai-structured-outputs
- *Testing Structured Outputs from an LLM*（分层错误类型与截断检测）— https://codoid.com/?p=41979

---

## 7. 可商用轻量量化系统架构规范

- 分层清晰：`data → features → model → backtest → report`，层间通过**文件/标准结构**交互；
- 无状态函数优先，避免深层继承；
- 可复现：固定随机种子、记录参数、记录数据版本；
- 全部计算可重跑，且重跑结果一致；
- 协议先行：只用 MIT / Apache-2.0 / BSD，避免 GPL 传染。

→ 落到本项目：四层固定、`store.py` 为唯一 I/O 边界、模型实现可替换、种子固定、`audit.log` 记录每次产物。

参考：
- Quant Research Lab（"文件即接口"、管道化）— https://github.com/YangJinYanXian/Quant-Research-Lab
- *Building a Clean, Scalable Quant Research Pipeline*（schema 校验、Parquet、避免前视）— https://www.metteyyaanalytics.com/blog/scalable-quant-research-pipeline-in-python
- *FactorForge*（"每个阶段读写文件"作为可复现保证）— https://www.pankajsomkuwarai.com/copy-of-voiceai

---

## 8. 无数据库、文件式轻量数据管线

- 小数据量（MB 级）用 CSV/Parquet 足够，且能 git diff、人工核对；
- 时间对齐是最大坑：统一时间口径（本项目统一为 `YYYY-MM` 月序号整数），缺失用显式标记而非静默填充；
- 写入要原子（临时文件 + rename），覆盖前备份；
- 增量更新要能回答"这条数据哪来的、什么时候来的"。

→ 落到本项目：`store.write_csv/json` 原子写 + `.bak`；`audit.log` 追加式记录；月度缺口超过 2 个月不填充而是告警；因子长表保留 `source_url` / `evidence`。

参考：
- aquant 架构解析（Parquet 落地、时间对齐、复权口径）— https://www.csdn.net/detail/webblog_27422290/21917
- Quantex（CSV/Parquet 数据源抽象、事件驱动）— https://mrgreeny.dev/projects/quant-finance-backtesting-library

---

## 9. 大模型 API / 手动双模式兼容架构

双模式的本质不是"两套代码"，而是**一份契约、两个传输通道**：

| 必须统一 | 允许不同 |
|---|---|
| 提示词模板（含占位符与自检清单） | 调用方式（HTTP / 人工复制） |
| JSON Schema 与版本号 | 模型与温度 |
| 校验器与质量门槛 | 重试策略 |
| 入库结构与去重规则 | 元数据里的 `mode` 字段 |

工程上加一道**模板指纹**（SHA256）：导入时重算并比对，模板改过就拒绝复用旧结果——这是防止"提示词升级了但结果还是旧的"最省事的办法。

→ 落到本项目：`prompt_builder.render()` 产出 system/user/hash；`extract.import_payload()` 与 `client.call_chat()` 汇入同一个 `ingest()`。

---

## 10. 经济供需因子 + 时序模型融合

文献里主流是**两段式残差修正**：

1. 时序/计量模型提取线性结构与季节（ARDL、ETS、Prophet）；
2. GBM 或专家网络对残差做非线性修正（ARDL-LightGBM、Prophet-SRC、Ridge+LSTM-Forest）。

另一条线是**结构弹性**：把 `log P` 回归到供给/需求指数 + 月份虚拟变量，得到可解释的 β，用于情景换算。

→ 落到本项目：
- 点预测用三模型集成（时序 + 树 + 基准）；
- 情景不用树（无法外推），改用弹性：`Δlog P = β_s·ΔS + β_d·ΔD`；
- 校验层强制 `β_s<0<β_d`，方向不对直接报 error；
- 需求类与进口类因子不进"供给松紧指数"——它们与价格同期内生，放进去会把内生性灌进弹性估计（本项目实测 β_s 会翻正）。

参考：
- *AI-Driven Supply Chain Demand Forecasting Using Hybrid Time-Series and Gradient Boosting*（时序 + 残差修正）— https://ijsred.com/volume9/issue2/paper-details/IJSRED-V9I2P147.html
- *Hybrid ARDL-LightGBM*（两段式残差学习，方向准确率为首要指标）— https://tdr.lib.ntu.edu.tw/jspui/handle/123456789/102701?mode=full
- *GPF-EVMoLE*（ETS 基准 + 变量选择 + 残差专家，明确"结构基线 vs 残差修正"分工）— https://dx.doi.org/10.3390/su18094404

---

## 11. 开源项目参考与协议核验

| 项目 | 协议 | 做了什么 | 本项目借鉴 / 未照搬 |
|---|---|---|---|
| [christoverhendri/Commodity-Data-Analysis](https://github.com/christoverhendri/Commodity-Data-Analysis) | 未声明（仅参考思路） | ARIMA/SARIMAX + XGBoost/RF/LightGBM 算术平均集成 | 借鉴"多模型 + 集成"骨架；**未照搬**其算术平均（改用反比 MSE，并按时间前推取权重） |
| [DhrumiS04/AgriPredict](https://github.com/DhrumiS04/AgriPredict) | Apache-2.0 | LSTM + SARIMA + ETS + NLP 情绪 | 借鉴"统计 + 学习模型互补"；**未照搬** LSTM（样本量不足）与情绪因子（不可稳定量化） |
| [PavithaDissanayake/VegetablePriceForecast](https://patch-diff.githubusercontent.com/PavithaDissanayake/VegetablePriceForecast) | 未声明（仅参考） | 采集→清洗→ARIMA/LSTM→Streamlit | 借鉴"采集-清洗-预测-展示"分层；**未照搬** Streamlit（要求单文件 HTML + 指定设计系统） |
| [YangJinYanXian/Quant-Research-Lab](https://github.com/YangJinYanXian/Quant-Research-Lab) | 见仓库 | "文件即接口"的轻量研究框架 | 直接采用该原则 |
| [akfamily/akshare](https://github.com/akfamily/akshare) | **MIT** | 财经数据接口库 | 作为可选行情适配器；**不改源码**，仅调用 `futures_zh_daily_sina` 并聚合 |
| [statsmodels](https://www.statsmodels.org/) | **BSD-3-Clause** | 计量/时序模型 | 可选对照实现（`fit_with_statsmodels`） |
| [microsoft/LightGBM](https://github.com/microsoft/LightGBM) | **MIT** | 梯度提升 | 可选后端 |
| [microsoft/qlib](https://github.com/microsoft/qlib) | **MIT** | AI 量化研究平台 | 仅参考工程分层，未引入（依赖过重） |
| [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | **MIT** | 金融大模型与数据 | 未引入（本项目不需要微调模型） |
| [AI4Finance-Foundation/FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | **Apache-2.0** | 金融 Agent 平台 | 未引入（Agent 编排超出范围） |
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | **Apache-2.0** | 多智能体交易研究 | 未引入（与"LLM 不预测价格"的设计冲突） |

**规避清单**：Backtrader（GPL-3.0）、PyQt/PySide（GPL/LGPL）等强传染协议一律不引入；本项目 UI 用原生 HTML，回测自研。

**协议核验的边界**（必须写在文档里，避免误读）：
- 仓库 LICENSE 说的是**代码**的授权；
- AKShare 抓取的行情来自交易所与资讯站点，**数据本身的商用授权需逐源核验**；
- 本项目不随附任何真实行情，`data/` 由脚本生成合成数据。

---

## 12. 从调研到实现的映射表

| 调研结论 | 代码位置 | 验证 |
|---|---|---|
| 收获窗口而非单点 | `core/timeutil.harvest_window` | `test_harvest_window_*` |
| 基差特征 | `data/features.build_dataset`（`basis`） | `TestFeatures` |
| 防未来函数 | `features._factor_monthly` / `_expanding_z` | `test_future_price_does_not_change_past_features` |
| 去趋势后再标准化 | `features._detrend_ratio` | 弹性符号测试 |
| 直接多步 | `models/gbm.py` + `features.build_targets` | `test_gbm_learns_supply_relation` |
| 留出验证定区间 | `gbm.py` 中 75/25 切分 | 回测覆盖率指标 |
| 反比 MSE + 时间前推权重 | `models/ensemble.py` + `backtest.py` | `test_rolling_backtest_metrics` |
| 弹性做情景 | `econ/elasticity.py` | `test_elasticity_sign` |
| 九项校验 | `econ/validator.py` | `TestEconValidation` |
| strict schema | `prompts/schema.json` + `jsonschema_lite` | `TestLLMSchema` |
| 双模式同契约 | `llm/prompt_builder.py` + `extract.ingest` | `test_prompt_render_has_no_placeholder_left` |
