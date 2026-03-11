# Quantitative-Forecasting-of-Ferrous-Production-A-Time-Series-Stacking-Approach-
# 量化研究项目：黑色系大宗商品日均铁水产量预测系统
基于 Stacking 集成学习的黑色系铁水产量量化预测模型
针对宏观基本面时间序列数据“样本极小、噪音极高”的痛点，本项目以全国日均铁水产量为核心标的，构建了一套深度契合产业物理周期的量化预测流水线。
项目首先重构了数据预处理逻辑，通过前向填充（ffill）彻底消除了时序数据中的未来函数（Look-ahead Bias）；在特征层面，严格锚定高炉 35 天复产周期构建滞后因子，并基于残差分布引入 Huber 权重对单日极端异常波动进行降权。
算法层面，自主搭建了 Time-Series Stacking 集成架构，底层并联 XGBoost（捕捉局部非线性冲击）与 Ridge 岭回归（锁定长期利润驱动底色）。最终模型成功克服了单一树模型在小样本下的过拟合瓶颈，测试集 MSE 降至 0.171，R² 实现跨越式转正（0.24）。经 XGBoost 底层归因分析，模型决策权重完美对齐“盈利驱动生产”的宏观经济逻辑，具备极高的实盘投研参考价值。

编程语言：Python

核心算法框架：Scikit-learn (StackingRegressor, Ridge, StandardScaler), XGBoost (Gradient Boosting Trees)

数据处理与分析：Pandas, NumPy

时序特征工程：Lagged Features (滞后特征对齐), Rolling Windows (滚动时间窗口), Forward Fill (前向填充防数据泄露)

模型评估与归因：MSE (均方误差), R² (决定系数), Multi-metric Feature Attribution (基于 Gain/Weight/Cover 的多维特征归因)

鲁棒性优化：Huber-style Sample Weighting (基于残差的异常值动态降权)
###### 为什么有些没利润，有些铁水产量和钢业企业是空的？

<img width="633" height="502" alt="image" src="https://github.com/user-attachments/assets/6f635963-f412-4602-992c-05101a2f703d" />


### 第一步：看数据（Data）—— 搞清楚“因”和“果”

量化第一步是看数据。你先打开 `data_input/日均铁水产量.xlsx`：

- **“果”（Label）**：日均铁水产量（这是你要预测的目标）。
- **“因”（Features）**：螺纹钢利润、高炉盈利率、基差、库存等。
- **业务逻辑**：钢厂赚不赚钱（利润）、厂里货多不多（库存），直接决定了它明天开不开工。

我的分析和报告：

**越往后，数据越多**

这个时间点之前都是周报，然后后年就是每天有数据了，都有利润，但是日均铁水产量和247家钢铁企业盈利率从9-13开始没7天出现一次，螺纹盘面利润从13开始每天都有了，热卷厂内/社会库存比值从2015.5.8开始有了，2016.8.5开始有了五大钢材周度表需(预测/3年季节性/20期)2017-11-28 00:00:00开始有了螺纹高炉利润/上海/即期和热卷高炉利润/上海/即期数据，但是中间有一些日期如2020 1.24，1.31，6.26缺失，然后过了一段时间才有了日均铁水产量模拟值(预测/装置跟踪)，最后最近的一段时间又只有日均铁水产量模拟值(预测/装置跟踪)和五大钢材周度表需(预测/3年季节性/20期)的数据了，铁矿基差率和焦炭基差率也在较后期出现

2017-11-28:即期利润出现 即期利润 = 1吨螺纹钢的价格 - (1.6吨铁矿石价格 + 0.5吨焦炭价格 + 其他辅材和加工费)

模拟值和基差出现：说明行业进入预测时代和期现结合时代

2020-1月底的缺失可能是疫情，所以不要用线性插值

## 初始数据

------

<img width="644" height="265" alt="image" src="https://github.com/user-attachments/assets/d8f5386a-d3bf-405b-8d89-3ee6e6d43b3d" />


<img width="641" height="434" alt="image" src="https://github.com/user-attachments/assets/5d09097d-beae-483e-b82f-f73389d9feaf" />


gain 含金量 对mse的贡献

cover 责任面 cover高，代表指标对大多数情况管用，低，泛化差

total 就是乘以weight对总贡献，总覆盖度

**你看那个1.000的gain，就是过度依赖了**

## 优化

1. #### 1. 修改填充逻辑（防止“作弊”）

   **原代码：** `fill_methods` 里的 `'interpolate'`。 **修改建议：** 全部改为 `'ffill'`（前向填充）。

   - **理由**：铁水产量是周更数据。线性插值会让你在周一就“偷看”了周五的答案（未来函数）。改用 `ffill` 能确保预测周一到周四时，只使用已知的上周五数据，这才是真实的实盘逻辑。

<img width="612" height="219" alt="image" src="https://github.com/user-attachments/assets/55da6fd3-a4d0-4851-8ee1-2e3838fa0e1f" />


<img width="618" height="433" alt="image" src="https://github.com/user-attachments/assets/b43d7323-8e3c-4067-84a8-11a308af0a04" />


   

   我的总结：

   就是训练集mse虽然高了，但是预测曲线波动性更加好，一开始线性插值所以平滑，现在敏感度保留了，一开始能看到未来数据就是作弊啊，ffill用上一个有效值填充

   测试集MSE从从 **0.8055** 降到了 **0.5013**

   35天那个指标和日均铁水产量模拟值的gain相对上升了，不迷信指标了

   日均铁水产量模拟值的weight在第一了，模型更多利用模拟盘，更专业

   **为什么特征工程做对了**

   在原始版本中，模型过度依赖插值后的盈利率因子（Gain 过于集中）。优化为 ffill 后，**即期利润因子的 Total_Gain 显著提升**，这说明模型开始真正捕捉高频的价格传导信号，而不是单纯拟合平滑后的曲线。因为提前35天的total_cover提升了

   #### 2. 优化滞后参数（捕捉更快的反应，失败了哈，负面优化）

   **原代码：** `shift(35)` 或 `shift(28)`。 **修改建议：** 尝试将 35 改为 **21** 或 **14**。

   - **理由**：你观察到 2017 年后数据更精密了。现代钢铁行业的反馈周期在变快，缩短滞后天数可能让模型对近两年的行情更灵敏。

   ### 报错和解决

<img width="619" height="348" alt="image" src="https://github.com/user-attachments/assets/62dec626-9b8d-4db9-b62d-5e61530c1438" />


   其实只是不小心删了一行。

   结果mse高了10倍

   #### 3. 扩大样本容量（增强泛化能力）

   **原代码：** `sheet_daily['Date'] >= pd.Timestamp('2024-02-20')`。 **修改建议：** 改为 `2023-01-01` 或更早。

   - **理由**：原代码只用了不到一年的数据，样本太少，AI 很容易“死记硬背”这一年的走势（过拟合）。增加一年数据能让模型见识更多的市场波动。

   ------

<img width="611" height="346" alt="image" src="https://github.com/user-attachments/assets/26768f0c-0910-4585-90d4-7b67365ab795" />


也高了，排除

## 调参

**调整 `learning_rate`**： 把 `0.09` 改成 `0.05`，同时把 `num_boost_round` 增加到 `10000`。

- **理由**：步子迈小一点，配合 `early_stopping`，可以让模型找的最优点更精准。

**调整 `max_depth`**： 从 `7` 降到 `5`。（就这个有用）

- **理由**：铁水数据特征不多（只有 7 个），树太深容易“钻牛角尖”，降低深度能进一步防止过拟合。

<img width="640" height="317" alt="image" src="https://github.com/user-attachments/assets/42d47079-eaf6-4ec2-865b-773f6e6f8760" />


<img width="648" height="457" alt="image" src="https://github.com/user-attachments/assets/8f8df392-76ef-4339-942a-4af6f1bd3cbd" />


- **原理解析**：在小样本数据集上，深层树（Depth=7）就像是一个“显微镜”，它会把数据里的随机噪音（比如某次调研的统计误差）当成真理去记。
- **优化效果**：你把深度降到 5 或 6，相当于给模型加了一个**“磨砂滤镜”**，强迫它忽略细枝末节，只看利润和产量的主要矛盾。这就是为什么 MSE 能降到 **0.4079**。

#### 换随机森林和ridge回归

<img width="640" height="508" alt="image" src="https://github.com/user-attachments/assets/21127b34-ae5d-493a-96b4-d1ef2ede13ab" />


<img width="637" height="474" alt="image" src="https://github.com/user-attachments/assets/e5c5d2b2-08a5-487c-9589-e0886a0535aa" />


性能都没有提升

## 使用多层 Stacking 集成学习架构

<img width="649" height="678" alt="image" src="https://github.com/user-attachments/assets/5f57dec1-b7ae-4f0b-b5c9-801555fe0ba9" />


**第一层 (Base Learners)**：融合了调优后的 **XGBoost**（捕捉非线性产业波动）与 **Ridge**（锁定线性利润驱动逻辑）。

**第二层 (Meta-Learner)**：使用简单的 **Ridge 回归** 作为元模型，通过 5 折交叉验证（Cross-Validation）生成的预测值进行二次拟合。

#### 第一步：数据清洗与时序防作弊 (Data Preprocessing)

> “拿到数据后，我首先统一了时间索引。在处理缺失值时，我发现原代码使用了 `interpolate`（线性插值）。这在时间序列预测中会引入严重的**未来函数**（Data Leakage）。我将其全部重构为 **`ffill`（前向填充）**，确保模型在训练时只能使用‘历史已知’的数据，完全贴合实盘交易和预测的真实环境。”

#### 第二步：基于产业逻辑的特征工程 (Feature Engineering)

> “在因子处理上，我严格保留了滞后项（Lagging Features），比如将螺纹利润前置 35 天。我做过灵敏度测试，试图缩短这个周期，但发现误差反而变大。这从数据层面印证了黑色产业链中，从高炉复产意愿到实际铁水产出，存在一条不可违背的**物理时间差**。同时，我加入了核心因子的滚动均值，帮助模型捕捉短期动量。”

#### 第三步：引入 Huber Loss 防御噪音 (Robust Weights)

> “钢铁调研数据样本少（仅一年多），且极易受到宏观情绪引发的单日异常波动影响。为了防止模型过拟合这些‘脏数据’，我在训练前计算了目标值的残差分布，并借鉴了 **Huber Loss** 的思想，对偏离度超过 90% 分位数的异常样本进行了**动态降权（Down-weighting）**。这构成了模型第一层强大的防线。”

#### 第四步：构建多层 Stacking 集成架构 (Model Architecture)

> “针对极小样本的建模，单一的树模型容易陷入局部过拟合（表现为 $R^2$ 极不稳定）。为此，我搭建了 **Stacking 集成框架**：
>
> - **底层 (Level-0)**：使用调优后的 **XGBoost**（限制 max_depth=5 抑制复杂度）捕捉非线性冲击，同时并联 **Ridge 岭回归** 锁定利润与产量之间的长周期线性底色。
> - **顶层 (Level-1)**：利用交叉验证（CV）将两者的输出馈送给元模型，实现了方差与偏差的极致平衡。最终使测试集 MSE 下降至 0.17 级别。”

加了两个特征没啥用



<img width="648" height="529" alt="image" src="https://github.com/user-attachments/assets/5c738284-f026-4ba2-b942-cdda79d53bb3" />


<img width="646" height="405" alt="image" src="https://github.com/user-attachments/assets/3e8dd33b-a383-4e71-a20c-f2a5825de71e" />


### 📊 这张表的最终解读（发邮件前必看）

你把这张图放进 Word 里时，可以配上这段**“点睛之笔”**，展现你对数据背后业务逻辑的理解：

- **盈利率是核心驱动力**：在 `gain` 和 `total_gain` 上都是唯一的 **1.0000**，这证明了模型逻辑非常扎实——钢厂只有在盈利的情况下才会有动力维持或增加产量。
- **模拟值是基础支撑**：虽然其单次分裂的增益（gain）不如盈利率，但其 `weight` 是 **1.0000**，说明模型在构建每一棵决策树时，都高频参考了这个物理跟踪指标，它起到了预测的“基准锚点”作用。
- **库存与利润的协同**：螺纹利润和库存比值的 `total_gain` 紧随其后，形成了一个完整的**“盈利-库存-生产”**逻辑闭环。
