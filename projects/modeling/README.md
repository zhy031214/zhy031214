# 电商会话购买意图预测

> 用可解释机器学习和倾向得分因果分析，识别高购买概率会话，并单独估计「周末访问」的因果效应。

[← 返回项目案例](../10-purchase-intent-modeling.md) · [← 返回主页](../../README.md)

## 核心结果

| 指标 | 结果 |
|---|---:|
| 会话数 | 12,330 |
| 购买率 | 15.47% |
| 最终模型 | RandomForest |
| 测试集 PR-AUC | 0.7290 |
| 测试集 ROC-AUC | 0.9199 |
| 阈值（训练集折外 F1 最优） | 0.415 |
| Precision / Recall / F1 | 0.6606 / 0.6623 / 0.6614 |
| AIPW 周末效应 | +1.59 个百分点，95% CI [+0.03%, +3.10%] |

## 项目问题

电商平台每天产生大量会话。如果能在会话早期识别高购买概率访问，就可以把优惠券、客服和再营销资源投向更可能转化的人群。

项目区分两个不同问题：

1. **预测问题**：根据会话特征预测 `Revenue=1` 的概率。
2. **因果问题**：在观察数据和明确识别假设下，估计周末访问相对工作日访问对购买率的平均影响。

预测不等于因果。置换重要性和 SHAP 只解释模型；因果部分单独使用倾向得分、IPW 和 AIPW。

## 方法

- 数据审计：缺失、重复、类型、目标比例。
- EDA：月份、访客类型、工作日/周末、PageValues 等业务分组。
- 特征工程：`TotalPages`、`TotalDuration`、`AvgSecondsPerPage`、`ProductPageShare`、`PageValueSeen`。
- 预处理：`Pipeline` + `ColumnTransformer`，所有拟合只在训练折内完成。
- 模型：逻辑回归、决策树、随机森林、HistGradientBoosting；五折分层交叉验证，以 PR-AUC 调参。
- 阈值：用训练集折外预测选择 F1 最优阈值 0.415，避免使用测试标签。
- 解释：置换重要性、逻辑回归系数、SHAP；明确其边界是「解释模型」，不是「证明因果」。
- 因果：倾向得分、IPW、AIPW、SMD 平衡、200 次自助法置信区间。
- 稳健性：移除 `PageValues` 后重新训练，报告性能变化。

## 目录

```text
modeling/
├── code/analysis.py          # 完整分析脚本
├── data/README.md            # 数据来源与下载方式（不直接分发原始数据）
├── results/                  # 最终指标与因果估计
├── requirements.txt
└── README.md
```

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 从 UCI 下载数据，放到 data/online_shoppers_intention.csv
python code/analysis.py
```

数据下载与字段说明见 [`data/README.md`](./data/README.md)。

## 数据引用

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). CC BY 4.0.

## 说明

- 原始数据不随仓库分发，请从 UCI 下载。
- 项目代码用于作品集展示；如果你要引用或复用，请先联系我。
- 完整图表、报告和面试讲解材料在本地维护，案例页里有核心图表与结论。
