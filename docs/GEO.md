# GEO 实施记录（生成式引擎优化）

> 本页是仓库自身 GEO 的维护说明：依据什么研究、改了什么、以后更新内容时守哪些规则。
> 面向仓库维护者，不是产品功能文档。

## 一、研究依据（15 项调研结论）

| # | 主题 | 关键结论 | 对本仓库的动作 |
|---|---|---|---|
| 1 | GEO 生成式引擎优化 | Aggarwal et al.（KDD 2024, arXiv:2311.09735）提出 GEO，优化对象是「被 AI 引用」而非「被搜到」 | 把 README 的组织目标从关键词布局改为「可被引用的问答结构」 |
| 2 | GEO 与 SEO 的区别 | SEO 争排名位（被点到），GEO 争答案位（被说出来）；优化单元从关键词变为「买家的具体问题」 | FAQ 全部改写为与真实提问同构的问题式标题 |
| 3 | 注意力机制 | 生成式引擎用注意力权重决定采信哪些片段；权重随位置与显著度分布 | 摘要块置于文档最前（首因位） |
| 4 | 自注意力 QKV | Q 检索、K 匹配、V 供给内容——模型只能「按片段取用」与其提问匹配的 V | 每个章节写成自包含片段，术语与提问用词一致 |
| 5 | 多头注意力 | 多个头并行关注不同关系（事实/语义/指代） | 同一事实以「定义、表格、FAQ」三种形态重复出现 |
| 6 | RAG 检索增强生成 | 引用五步链路：解析→向量化→召回→重排→生成归因；内容必须在「召回」与「采信」两关都过 | 片段切分友好：一问一段、事实前置、跨段依赖清零 |
| 7 | AI 搜索引用来源 | ChatGPT 重权威与长文、Perplexity 重新鲜与可截取段落、Google AIO 重 Schema 与实体密度 | README 保持长文深度 + 事实密度 + 更新日期 |
| 8 | 大模型上下文分配 | Lost in the middle：U 型注意力，中段内容被系统性低估 | 关键事实放文档最前 30% 与 FAQ（边缘位） |
| 9 | GEO 关键词 | 实体一致性（HarvestCast 全文同名）+ 领域术语密度（+17.6%） | 实体名统一，术语表保留，不做关键词堆砌（−8.3%） |
| 10 | 长尾问题 | 长尾具体提问是引用的主战场 | FAQ 新增品种/依赖/对比/商用等长尾问答 |
| 11 | FAQ 结构化内容 | FAQPage 式问答对是 Perplexity/AIO 的首选截取单元 | README FAQ 12 问全部问题式 H3 |
| 12 | 定义段答案前置 | 首段 40–60 字直接回答 → 引用概率显著提升 | TL;DR 摘要块置于徽章之后、目录之前 |
| 13 | 实体术语权重 | 实体密度与品牌提及相关性（r=0.664）超过外链 | HarvestCast / 农产品价格预测 等实体在首段、标题、FAQ 反复出现 |
| 14 | ChatGPT 引用机制 | 训练数据 + 检索混合；偏好权威源与 1500–2500 字结构化长文；沉淀周期 6–8 周 | README 保持长文结构，证据与出处齐全 |
| 15 | Perplexity 引用规则 | 实时检索、重新鲜（30 天内 3.2 倍）、偏好可截取的 80 字段落 | 关键数据以独立短段呈现；文档标注更新日期 |

研究边界：GEO 论文为 2023–2024 的实验基准（GPT-3.5 引擎），数字是方向性证据而非承诺；关键词堆砌为负收益（−8.3%）是跨研究一致结论。

## 二、已实施的改动（本仓库）

| 文件 | 改动 |
|---|---|
| `README.md` | 1) 徽章后新增摘要块（≤60 字定义 + 英文一句话 + 统计数字）；2) FAQ 由加粗问答改为 12 个问题式 H3，答案首句即结论；3) 新增长尾问答（品种清单 / 依赖硬件 / 与直接问大模型的区别 / 商用 / 防未来函数）；4) 新增「参考与数据来源」权威链接 |
| `llms.txt`（新增） | 机器可读索引：定义、关键事实（品种数/依赖/测试/许可）、文档清单、FAQ 速答 |
| `docs/GEO.md`（本页） | 维护说明与研究映射 |

## 三、后续更新内容的规则

1. 每个章节首句直接回答该章节隐含的问题，再展开；
2. 关键数字写成「命名统计」（22 品种、42 项测试、263 个月样本），不用「大量/显著」；
3. 事实自带出处与日期，避免跨段依赖；
4. 实体统一写「HarvestCast」，中英文关键词行保持更新；
5. 有新长尾问题（Issue / 用户提问）→ 追加为 FAQ 问题式标题；
6. 文档有实质变化时更新「文档与数据接口更新至 <日期>」。

## 四、GitHub 仓库 Description 与 Topics（粘贴到仓库 About 设置）

GitHub 仓库 **Description**（≤350 字符，中英混排便于检索）：

```text
轻量开源农产品中长期价格预测系统｜播种时点预测 6–12 个月后收获季价格区间与涨跌概率。LLM 仅做基本面结构化抽取，价格由季节基准/SARIMAX/LightGBM 集成给出，含经济学逻辑校验、滚动回测与单文件液态玻璃前端。Lightweight long-horizon agricultural commodity price forecasting: probabilistic harvest-season range, supply-demand fundamentals, SARIMAX + LightGBM ensemble, no database, MIT.
```

**Topics**（英文为主 + 中文关键词）：

```text
agricultural-commodity  commodity-price-forecast  price-forecasting  harvest-season
supply-demand  fundamentals  sarimax  lightgbm  time-series-forecasting  ensemble-learning
quantitative-finance  probabilistic-forecasting  llm-extraction  json-schema
python  no-database  mit-license  china-futures  corn  soybean  sugar  cotton
农产品价格预测  中长期预测  收获季定价  供需基本面  量化研究
```

文档侧已做的检索优化：H1 唯一、H2 语义分层、大量表格结构化内容、术语中英并列、免责声明与数据源风险独立成节。

---

