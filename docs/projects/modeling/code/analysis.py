# %% [markdown]
# # 电商会话购买意图预测：从数据审计到因果解释
#
# 本项目使用 UCI Online Shoppers Purchasing Intention 数据集，预测一次电商访问是否会以购买结束，
# 并单独估计“周末访问”对购买率的因果效应。
#
# **阅读约定**
#
# - 预测问题：在已知一次会话的行为特征后，识别高购买概率会话。这里只能说“预测相关”，不能说某个特征一定导致购买。
# - 因果问题：在一致性、条件可交换性、正性和 SUTVA 等假设下，估计周末与工作日之间的平均购买率差异。
# - 每段代码前均说明输入、输出、方法和不能证明的结论，方便逐块复现和面试讲解。

# %%
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import joblib
import matplotlib

# 脚本模式使用无界面的 Agg 后端；Notebook 保留 Jupyter 的 inline 后端，
# 这样 plt.show() 既能保存图片，也能把图片嵌入 Notebook 单元输出。
try:
    get_ipython()  # type: ignore[name-defined]
    IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False
if not IS_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore")

# 固定随机种子，保证数据划分、模型训练和自助法结果可复现。
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 同时兼容“直接运行 .py”和“在项目根目录执行 Notebook”两种情况。
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
else:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULT_DIR = PROJECT_ROOT / "results"
for folder in (DATA_DIR, FIGURE_DIR, RESULT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# 中文字体按优先级查找；必须放在 sns.set_theme 之后，否则主题会覆盖字体设置。
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Hiragino Sans GB",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

print(f"项目目录：{PROJECT_ROOT}")

# %% [markdown]
# ## 0. 方法因果索引
#
# 下表的“因果链”是指方法如何把输入转化为结果、结果又怎样进入下一步分析；
# 它不是统计学意义上的因果证明。真正的因果估计只在最后单独设计的周末效应实验中出现。

# %%
METHOD_CHAINS = [
    {
        "阶段": "数据审计",
        "方法": "缺失值、重复值与类型检查",
        "为什么用": "先确认数据是否具备分析资格",
        "输入→机制": "逐列检查缺失、类型、取值范围和完全重复行，识别异常会影响均值、编码和模型训练",
        "输出→下一步": "决定是否删除记录、转换类型或保留原样；本项目保留重复会话并记录原因",
        "可以下的结论": "数据完整性、重复情况与字段类型",
        "不能证明": "字段一定准确，或重复记录一定是错误",
    },
    {
        "阶段": "数据划分",
        "方法": "分层训练/测试集划分",
        "为什么用": "模拟模型遇到未见会话时的表现",
        "输入→机制": "按购买标签比例分别从总体抽取训练集和测试集",
        "输出→下一步": "训练集用于调参，测试集只用于最终评估",
        "可以下的结论": "模型对同类未参与训练会话的泛化表现",
        "不能证明": "模型能外推到其他网站、国家或未来时期",
    },
    {
        "阶段": "特征工程",
        "方法": "页面总数、总停留时间、单页时长等组合特征",
        "为什么用": "把分散的行为记录压缩为可解释的参与强度",
        "输入→机制": "对页面数量和时长做加减除法，得到浏览深度与平均停留",
        "输出→下一步": "进入模型，检验会话参与度是否提高购买概率的预测信息",
        "可以下的结论": "组合特征是否改善泛化指标",
        "不能证明": "增加停留时间就会直接导致购买；行为可能是购买意愿的结果",
    },
    {
        "阶段": "预处理",
        "方法": "中位数/众数填充",
        "为什么用": "让流水线能够处理未来可能出现的缺失值",
        "输入→机制": "在训练集学习数值中位数和类别众数，再填充新数据",
        "输出→下一步": "形成无缺失的建模矩阵",
        "可以下的结论": "缺失不会导致训练报错",
        "不能证明": "填充值等于用户真实但未记录的行为",
    },
    {
        "阶段": "预处理",
        "方法": "独热编码",
        "为什么用": "避免把浏览器、地区等编号误当成大小顺序",
        "输入→机制": "每个类别生成 0/1 指示列，删除第一个类别作为参照",
        "输出→下一步": "供线性模型和树模型读取",
        "可以下的结论": "不同类别可被区分，模型不假设编号有序",
        "不能证明": "不同浏览器的效果是独立的，或类别编码本身有因果含义",
    },
    {
        "阶段": "预处理",
        "方法": "标准化",
        "为什么用": "让逻辑回归和倾向得分模型公平比较不同量纲的数值特征",
        "输入→机制": "使用训练集均值和标准差，把数值变为近似均值 0、方差 1",
        "输出→下一步": "降低停留秒数等大数值对系数求解的影响",
        "可以下的结论": "线性模型系数可比较",
        "不能证明": "标准化改变了用户行为或因果关系",
    },
    {
        "阶段": "不平衡处理",
        "方法": "类别权重",
        "为什么用": "避免大量“未购买”样本让模型只学会预测多数类",
        "输入→机制": "提高少数类样本在损失函数中的权重",
        "输出→下一步": "模型更关注购买样本，但概率校准可能改变",
        "可以下的结论": "模型在少数类上的召回率可能提高",
        "不能证明": "购买样本真的变多了，也不能保证 F1 一定提高",
    },
    {
        "阶段": "候选模型",
        "方法": "逻辑回归",
        "为什么用": "提供可解释的线性基准",
        "输入→机制": "用 sigmoid 把特征的线性组合映射为购买概率",
        "输出→下一步": "得到概率、系数和优势比",
        "可以下的结论": "在控制其他已入模特征时，特征与购买对数几率的关系",
        "不能证明": "系数就是因果效应；遗漏变量、共线性和反向影响都会破坏因果解释",
    },
    {
        "阶段": "候选模型",
        "方法": "决策树",
        "为什么用": "展示接近业务规则的非线性切分",
        "输入→机制": "每次选择使分类不纯度下降最多的阈值，递归形成树",
        "输出→下一步": "得到可读的规则和概率",
        "可以下的结论": "样本被哪些特征和阈值分组",
        "不能证明": "树的一次切分代表自然界的真实因果机制；单树通常不稳定",
    },
    {
        "阶段": "候选模型",
        "方法": "随机森林",
        "为什么用": "降低单棵树的方差，捕捉非线性与交互",
        "输入→机制": "多棵随机抽样特征与样本的树分别训练，再平均预测",
        "输出→下一步": "得到更稳定的集成概率",
        "可以下的结论": "模型能够利用非线性和交互关系提高预测",
        "不能证明": "特征重要性就是因果贡献",
    },
    {
        "阶段": "候选模型",
        "方法": "HistGradientBoosting（梯度提升树）",
        "为什么用": "通过梯度提升逐步修正残差，提升表格数据预测性能",
        "输入→机制": "按顺序增加弱分类树，每棵树学习当前模型的负梯度",
        "输出→下一步": "得到高精度概率，并进一步做 SHAP 解释",
        "可以下的结论": "模型在该数据分布下的预测表现和局部贡献",
        "不能证明": "高预测性能代表机制正确或特征具有因果作用",
    },
    {
        "阶段": "模型选择",
        "方法": "五折分层交叉验证",
        "为什么用": "在训练集内部评估调参后的泛化能力，减少偶然划分",
        "输入→机制": "把训练集分五份，每次四份训练、一份验证，并保持类别比例",
        "输出→下一步": "用平均 PR-AUC 选择模型与超参数",
        "可以下的结论": "训练数据内的稳定泛化表现",
        "不能证明": "验证结果等于最终业务收益",
    },
    {
        "阶段": "模型选择",
        "方法": "网格搜索",
        "为什么用": "在预先定义的合理范围内寻找泛化更好的超参数",
        "输入→机制": "逐一组合候选深度、叶节点大小、学习率等参数并交叉验证",
        "输出→下一步": "输出每组参数的验证分数和最佳参数",
        "可以下的结论": "在指定搜索空间和评分函数下的最优组合",
        "不能证明": "搜索空间外不存在更好模型",
    },
    {
        "阶段": "评估",
        "方法": "PR-AUC、ROC-AUC、F1、召回率",
        "为什么用": "购买样本只占约 15%，单看准确率会高估模型",
        "输入→机制": "比较不同阈值下的精确率、召回率与概率排序能力",
        "输出→下一步": "判断模型是否真正识别出少数类",
        "可以下的结论": "模型排序能力和不平衡类别下的综合表现",
        "不能证明": "统计指标直接等于营销收益；成本结构不同，最佳阈值也不同",
    },
    {
        "阶段": "阈值",
        "方法": "基于训练集折外预测选择阈值",
        "为什么用": "0.5 不一定是业务上最优的分界线",
        "输入→机制": "在训练集使用交叉验证生成未泄露的折外概率，选择 F1 最大阈值",
        "输出→下一步": "把概率转换为“是否标记高意向”",
        "可以下的结论": "以 F1 为目标时的训练内最优阈值",
        "不能证明": "该阈值在其他季节或真实营销成本下仍最优",
    },
    {
        "阶段": "解释",
        "方法": "原始变量置换重要性",
        "为什么用": "测量打乱某业务变量后模型性能下降多少",
        "输入→机制": "反复随机打乱测试集中的一个原始字段，观察 ROC-AUC 变化",
        "输出→下一步": "排序模型依赖的预测信息",
        "可以下的结论": "模型对该变量的依赖程度",
        "不能证明": "变量对购买有因果作用；相关代理变量会分走重要性",
    },
    {
        "阶段": "解释",
        "方法": "SHAP",
        "为什么用": "把单个会话的预测值拆成各特征的正负贡献",
        "输入→机制": "基于合作博弈的 Shapley 值，估计特征加入模型后对预测的边际影响",
        "输出→下一步": "给出全局排序和局部归因",
        "可以下的结论": "在当前模型内部，一个特征如何推高或压低预测",
        "不能证明": "说明用户行为本身造成购买；SHAP 解释的是模型，不是现实世界机制",
    },
    {
        "阶段": "因果分析",
        "方法": "倾向得分与逆概率加权（IPW）",
        "为什么用": "调整可观测混杂因素，估计周末访问的平均效应",
        "输入→机制": "先预测每个会话在周末发生的概率，再用概率倒数给样本加权，使两组协变量更可比",
        "输出→下一步": "得到周末组与工作日组的加权购买风险差",
        "可以下的结论": "在可交换性、正性等假设下，周末的平均因果效应估计",
        "不能证明": "未测量混杂不存在；也不能自动保证所有协变量平衡",
    },
    {
        "阶段": "因果分析",
        "方法": "增广逆概率加权（AIPW）",
        "为什么用": "同时使用倾向模型和结果模型，只要其中一个设定正确就能保持一致估计",
        "输入→机制": "结合两组的结局预测与逆概率残差修正",
        "输出→下一步": "得到更稳健的平均效应估计与置信区间",
        "可以下的结论": "在因果假设和模型设定假设下的双重稳健估计",
        "不能证明": "即使两个模型都错误、或存在强未测量混杂，结论仍成立",
    },
]

method_chain_df = pd.DataFrame(METHOD_CHAINS)
method_chain_df.to_csv(
    RESULT_DIR / "method_causal_chain.csv", index=False, encoding="utf-8-sig"
)
print(f"方法因果索引：{len(method_chain_df)} 条")

# %% [markdown]
# ## 1. 读取数据与字段字典
#
# 数据来自 UCI Machine Learning Repository，共 12,330 条“不同用户、一年内”的电商会话记录。
# `Revenue=True` 表示该次访问完成交易，是二分类目标。

# %%
DATA_PATH = DATA_DIR / "online_shoppers_intention.csv"
df_raw = pd.read_csv(DATA_PATH)

# 显式把布尔目标转为 0/1，避免后续某些模型对 bool 类型处理不一致。
df_raw["Revenue"] = df_raw["Revenue"].astype(int)

print(f"数据规模：{df_raw.shape[0]:,} 行 × {df_raw.shape[1]:,} 列")
print(df_raw.head().to_string())

# 数据字典同时保留业务含义和因果角色，提醒哪些字段是“会话中行为”。
data_dictionary = pd.DataFrame(
    [
        ("Administrative", "数值", "行政类页面访问次数", "会话行为"),
        ("Administrative_Duration", "数值", "行政类页面总停留秒数", "会话行为"),
        ("Informational", "数值", "信息类页面访问次数", "会话行为"),
        ("Informational_Duration", "数值", "信息类页面总停留秒数", "会话行为"),
        ("ProductRelated", "数值", "商品相关页面访问次数", "会话行为"),
        ("ProductRelated_Duration", "数值", "商品相关页面总停留秒数", "会话行为"),
        ("BounceRates", "数值", "入口页面直接离开的比例", "会话行为/导航质量"),
        ("ExitRates", "数值", "页面成为会话最后一页的比例", "会话行为/导航质量"),
        ("PageValues", "数值", "访问页面对完成交易的估计价值", "强预测指标/可能含转化信息"),
        ("SpecialDay", "数值", "距离特定节日的接近程度，0–1", "访问前情境"),
        ("Month", "类别", "访问月份", "访问前情境"),
        ("OperatingSystems", "类别编号", "操作系统编号", "访问前情境"),
        ("Browser", "类别编号", "浏览器编号", "访问前情境"),
        ("Region", "类别编号", "地区编号", "访问前情境"),
        ("TrafficType", "类别编号", "流量来源类型编号", "访问前情境/渠道"),
        ("VisitorType", "类别", "新访客、回访访客或其他", "访问前身份"),
        ("Weekend", "布尔", "是否在周末访问", "因果分析中的处理变量"),
        ("Revenue", "布尔", "会话是否完成交易", "目标变量"),
    ],
    columns=["字段", "类型", "含义", "分析角色"],
)
data_dictionary.to_csv(DATA_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")
data_dictionary

# %% [markdown]
# ## 2. 数据审计
#
# 这里先回答三个问题：有没有缺失？有没有重复？类别字段有哪些取值？
# 125 行完全重复并不等于“错误”：原始字段较粗，且数据已按不同用户/会话采集，
# 相同特征组合可能来自不同时间或用户。因此主分析保留全部记录，避免主观删除导致样本选择偏差。

# %%
audit = {
    "rows": int(df_raw.shape[0]),
    "columns": int(df_raw.shape[1]),
    "missing_cells": int(df_raw.isna().sum().sum()),
    "duplicate_rows": int(df_raw.duplicated().sum()),
    "positive_count": int(df_raw["Revenue"].sum()),
    "negative_count": int((df_raw["Revenue"] == 0).sum()),
    "positive_rate": float(df_raw["Revenue"].mean()),
}
audit_df = pd.DataFrame(
    {
        "检查项": ["记录数", "字段数", "缺失单元格", "完全重复行", "购买会话", "未购买会话", "购买率"],
        "结果": [
            audit["rows"],
            audit["columns"],
            audit["missing_cells"],
            audit["duplicate_rows"],
            audit["positive_count"],
            audit["negative_count"],
            f"{audit['positive_rate']:.2%}",
        ],
    }
)
audit_df

# %%
print("类别字段取值：")
for col in ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]:
    values = sorted(df_raw[col].astype(str).unique().tolist())
    print(f"- {col}: {len(values)} 个取值，例如 {values[:8]}")

# 保存审计结果，供报告构建脚本读取，避免报告数字与代码输出不一致。
with open(RESULT_DIR / "audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)

# %% [markdown]
# ## 3. 探索性数据分析
#
# EDA 的作用是先观察分布、不平衡程度和业务分组差异，再决定建模策略。
# 这一阶段的差异只是描述性关联，尚未调整混杂因素。

# %%
# 图 1：目标分布。购买约占 15.47%，因此“全预测为未购买”的准确率会达到 84.53%，
# 但完全没有识别购买者，所以后续必须重点看 PR-AUC、召回率和 F1。
counts = df_raw["Revenue"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(
    ["未购买", "购买"],
    [counts[0], counts[1]],
    color=["#8AA1B1", "#E76F51"],
)
for bar, value in zip(bars, [counts[0], counts[1]]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        f"{value:,}\n({value / len(df_raw):.2%})",
        ha="center",
        va="bottom",
        fontsize=11,
    )
ax.set_ylim(0, counts.max() * 1.18)
ax.set_ylabel("会话数")
ax.set_title("图 1  购买会话与未购买会话的数量分布")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_target_distribution.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# 图 2：月、访客类型、周末三个业务维度的购买率。
# 这些图回答“哪类时会更常购买”，但因为群体构成不同，不能直接当作因果效应。
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

month_rate = (
    df_raw.groupby("Month", observed=True)["Revenue"]
    .agg(["mean", "size"])
    .sort_values("mean", ascending=False)
)
axes[0].bar(month_rate.index, month_rate["mean"], color="#2A9D8F")
axes[0].set_title("按月份")
axes[0].set_ylabel("购买率")
axes[0].tick_params(axis="x", rotation=45)

visitor_rate = (
    df_raw.assign(访客类型=df_raw["VisitorType"].str.replace("_", " "))
    .groupby("访客类型", observed=True)["Revenue"]
    .mean()
    .sort_values(ascending=False)
)
axes[1].bar(visitor_rate.index, visitor_rate.values, color="#E9C46A")
axes[1].set_title("按访客类型")
axes[1].tick_params(axis="x", rotation=15)

weekend_rate = df_raw.assign(访问日=df_raw["Weekend"].map({0: "工作日", 1: "周末"}))
weekend_rate = weekend_rate.groupby("访问日", observed=True)["Revenue"].mean()
axes[2].bar(weekend_rate.index, weekend_rate.values, color=["#457B9D", "#E76F51"])
axes[2].set_title("按工作日/周末")

for ax in axes:
    ax.set_ylim(0, 0.30)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

fig.suptitle("图 2  不同业务分组的购买率（描述性关联）", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_business_segment_rates.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# 图 3：PageValues 的分布差异。PageValues 是强预测变量，但它的定义与交易价值有关，
# 因此需要做“去掉 PageValues 后性能是否明显下降”的稳健性检查。
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(
    data=df_raw,
    x="Revenue",
    y=np.log1p(df_raw["PageValues"]),
    palette=["#8AA1B1", "#E76F51"],
    ax=axes[0],
)
axes[0].set_xticklabels(["未购买", "购买"])
axes[0].set_xlabel("")
axes[0].set_ylabel("log(1 + PageValues)")
axes[0].set_title("PageValues 分布")

for label, color in [(0, "#8AA1B1"), (1, "#E76F51")]:
    values = df_raw.loc[df_raw["Revenue"] == label, "ExitRates"]
    axes[1].hist(values, bins=30, alpha=0.55, label="未购买" if label == 0 else "购买", color=color)
axes[1].set_xlabel("ExitRates")
axes[1].set_ylabel("会话数")
axes[1].set_title("ExitRates 分布")
axes[1].legend()

fig.suptitle("图 3  两个关键行为变量的类别分布", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_key_feature_distributions.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# 图 4：数值特征的 Spearman 相关矩阵。
# 使用 Spearman 而不是 Pearson，因为多个时长变量明显右偏。
# 相关矩阵只描述同步变化，不能说明方向性因果。
numeric_corr_cols = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Revenue",
]
corr = df_raw[numeric_corr_cols].corr(method="spearman")
plt.figure(figsize=(9, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8},
)
plt.title("图 4  数值变量与购买标签的 Spearman 相关矩阵")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_spearman_correlation.png", dpi=180, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. 特征工程
#
# 组合特征的设计原则是：每一步都能用业务语言解释，且绝不使用目标 `Revenue` 生成输入。
# - `TotalPages`：三类页面访问次数之和，表示浏览广度。
# - `TotalDuration`：三类页面停留时间之和，表示会话投入。
# - `AvgSecondsPerPage`：总时长/总页数，表示单页平均关注度。
# - `ProductPageShare`：商品页占全部页面的比例，表示商品探索集中度。
# - `PageValueSeen`：是否访问过有估计价值的页面，把连续值转换为易解释指示变量。

# %%
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """在不使用目标变量的前提下，生成可解释的会话级组合特征。"""
    out = data.copy()

    page_columns = ["Administrative", "Informational", "ProductRelated"]
    duration_columns = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]

    out["TotalPages"] = out[page_columns].sum(axis=1)
    out["TotalDuration"] = out[duration_columns].sum(axis=1)

    # 分母可能为 0，使用 np.where 明确返回 0，避免除零产生无穷大。
    out["AvgSecondsPerPage"] = np.where(
        out["TotalPages"] > 0, out["TotalDuration"] / out["TotalPages"], 0.0
    )
    out["ProductPageShare"] = np.where(
        out["TotalPages"] > 0, out["ProductRelated"] / out["TotalPages"], 0.0
    )
    out["PageValueSeen"] = (out["PageValues"] > 0).astype(int)
    return out


df = engineer_features(df_raw)

# 目标单独保存；所有输入字段都从 X 中取得，防止误把标签带进特征。
y = df["Revenue"].astype(int)
X = df.drop(columns=["Revenue"])

engineered_summary = X[["TotalPages", "TotalDuration", "AvgSecondsPerPage", "ProductPageShare"]].describe().T
engineered_summary

# 保存建模数据集：保留目标与组合特征，便于不运行 Notebook 也能审计每一行。
df.to_csv(DATA_DIR / "modeling_dataset.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ## 5. 训练/测试集与预处理
#
# **因果关系的第一步是避免信息泄漏。**
# 训练集负责学习参数和选择超参数，测试集只在最后评估一次。
# 分层抽样保持训练集和测试集中的购买比例一致，否则每次划分都可能得到差异很大的 PR-AUC。

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("训练集：", X_train.shape, "购买率：", f"{y_train.mean():.2%}")
print("测试集：", X_test.shape, "购买率：", f"{y_test.mean():.2%}")

# OperatingSystems、Browser、Region、TrafficType 的数字只是类别编号，
# 必须作为类别处理，不能让模型误解为“浏览器 10 大于浏览器 2”。
categorical_columns = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
numeric_columns = [col for col in X.columns if col not in categorical_columns]


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """构造只使用训练数据拟合的预处理步骤。"""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


preview_preprocessor = make_preprocessor(numeric_columns, categorical_columns)
preview_matrix = preview_preprocessor.fit_transform(X_train.head(5))
print("预处理后矩阵形状：", preview_matrix.shape)
print("原始字段数：", X_train.shape[1], "；独热编码后列数增加是因为类别被展开。")

# %% [markdown]
# ## 6. 候选模型与超参数搜索
#
# 模型选择统一使用 **五折分层交叉验证 + PR-AUC**：
# - 分层：每一折保持购买/未购买比例；
# - PR-AUC：关注少数类“购买”的排序质量；
# - 网格搜索：只在人工限定的合理参数范围内选模型。
#
# 训练集内选出的最佳参数随后固定，最终只在测试集报告一次。
#
# **为什么没有使用 LSTM？** 每行是一次独立会话，没有真实时间戳，也没有同一用户的事件序列。
# 若把相邻行强行当作时间序列，会让模型学习并不存在的时间先后，因此这里不构造伪序列。

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_pipeline(model) -> Pipeline:
    """把预处理和模型串成同一个 Pipeline，避免在新数据上重复手工处理。"""
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
            ("model", model),
        ]
    )


search_specs = {
    "LogisticRegression": {
        "estimator": LogisticRegression(
            C=0.1, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE
        ),
        "param_grid": {
            "model__C": [0.01, 0.05, 0.1, 0.5],
            "model__class_weight": [None, "balanced"],
        },
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [20, 100],
            "model__class_weight": [None, "balanced"],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(
            n_estimators=300, n_jobs=1, random_state=RANDOM_STATE
        ),
        "param_grid": {
            "model__max_depth": [8, None],
            "model__min_samples_leaf": [2, 10],
            "model__class_weight": [None, "balanced_subsample"],
        },
    },
    "HistGradientBoosting": {
        "estimator": HistGradientBoostingClassifier(
            max_iter=250,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
        "param_grid": {
            "model__max_leaf_nodes": [15, 31],
            "model__learning_rate": [0.05, 0.1],
            "model__class_weight": [None, "balanced"],
        },
    },
}

cv_rows = []
searches = {}

for model_name, spec in search_specs.items():
    print(f"\n正在训练并搜索：{model_name}")
    search = GridSearchCV(
        estimator=build_pipeline(spec["estimator"]),
        param_grid=spec["param_grid"],
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=1,
    )
    search.fit(X_train, y_train)
    searches[model_name] = search

    # 把每组超参数的验证结果保存下来，便于复查“为什么选这个参数”。
    for _, row in pd.DataFrame(search.cv_results_).iterrows():
        cv_rows.append(
            {
                "model": model_name,
                "params": str(row["params"]),
                "mean_cv_pr_auc": row["mean_test_score"],
                "std_cv_pr_auc": row["std_test_score"],
                "rank": row["rank_test_score"],
            }
        )
    print(f"最佳参数：{search.best_params_}")
    print(f"最佳 CV PR-AUC：{search.best_score_:.4f}")

search_results = pd.DataFrame(cv_rows).sort_values(["model", "rank"])
search_results.to_csv(RESULT_DIR / "model_search_results.csv", index=False, encoding="utf-8-sig")

# DummyClassifier 不使用任何业务特征，只预测训练集的多数类/先验概率，
# 它提供“如果不做数据挖掘会怎样”的最低基准。
dummy = DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)

models = {name: search.best_estimator_ for name, search in searches.items()}
models = {"DummyBaseline": dummy, **models}

# %% [markdown]
# ## 7. 测试集评估：为什么不能只看准确率
#
# 本次测试集中约 85% 是未购买。一个“全部判为未购买”的模型准确率约为 85%，但购买召回率为 0，
# 对营销几乎没有价值。因此同时报告：
# - ROC-AUC：概率排序能力；
# - PR-AUC：更关注少数类购买；
# - Precision / Recall / F1：在指定阈值下的表现；
# - 混淆矩阵：具体误报与漏报数量。

# %%
def threshold_metrics(y_true, y_pred):
    """计算固定阈值下可直接用于业务沟通的分类指标。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


test_probabilities = {}
metric_rows = []
for name, model in models.items():
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = threshold_metrics(y_test, prediction)
    metric_rows.append(
        {
            "model": name,
            "roc_auc": roc_auc_score(y_test, probability),
            "pr_auc": average_precision_score(y_test, probability),
            **metrics,
        }
    )
    test_probabilities[name] = probability

metric_df = pd.DataFrame(metric_rows).sort_values("pr_auc", ascending=False)
metric_df.to_csv(RESULT_DIR / "test_metrics_at_0_5.csv", index=False, encoding="utf-8-sig")
metric_df.round(4)

# %%
# 图 5：模型指标对比。PR-AUC 和 F1 是本项目的核心指标，准确率仅作辅助。
plot_metrics = metric_df.sort_values("pr_auc")
fig, ax = plt.subplots(figsize=(10, 4.8))
plot_metrics.set_index("model")[["pr_auc", "f1", "recall"]].plot(
    kind="barh", ax=ax, color=["#2A9D8F", "#E9C46A", "#E76F51"]
)
ax.set_xlabel("分数")
ax.set_ylabel("")
ax.set_title("图 5  测试集模型表现（阈值 0.5）")
ax.legend(["PR-AUC", "F1", "Recall"], loc="lower right")
ax.set_xlim(0, 1.0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_model_metrics.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# 图 6、图 7：ROC 曲线和 PR 曲线。
# ROC-AUC 衡量整体排序；PR-AUC 对少数类更敏感，因此本项目以 PR-AUC 作为主要选择依据。
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, probability in test_probabilities.items():
    if name == "DummyBaseline":
        continue
    fpr, tpr, _ = roc_curve(y_test, probability)
    ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, probability):.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="随机猜测")
ax.set_xlabel("假阳性率")
ax.set_ylabel("真阳性率")
ax.set_title("图 6  测试集 ROC 曲线")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_roc_curves.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 5.5))
for name, probability in test_probabilities.items():
    if name == "DummyBaseline":
        continue
    precision, recall, _ = precision_recall_curve(y_test, probability)
    ax.plot(recall, precision, label=f"{name} ({average_precision_score(y_test, probability):.3f})")
ax.axhline(y_test.mean(), linestyle="--", color="gray", label=f"正类比例 {y_test.mean():.3f}")
ax.set_xlabel("召回率")
ax.set_ylabel("精确率")
ax.set_title("图 7  测试集 Precision-Recall 曲线")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "07_precision_recall_curves.png", dpi=180, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. 阈值选择与最终分类
#
# **为什么不用 0.5？** 概率大于 0.5 只是常见默认值，不代表它最符合不平衡分类。
# 这里用训练集的折外预测选择 F1 最大的阈值，再把该阈值应用到完全未参与训练的测试集。
# 这样既能改变业务运营规模，又不会偷看测试标签。

# %%
# 只用训练集内部交叉验证的 PR-AUC 选择模型；测试集不参与任何模型或阈值选择。
best_model_name = max(searches, key=lambda name: searches[name].best_score_)
best_model = models[best_model_name]
print("最佳模型：", best_model_name)

# 用同一种五折方式得到训练集折外概率；这些概率没有见过对应折的真实标签。
train_oof_probability = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

threshold_grid = np.linspace(0.05, 0.95, 181)
threshold_scores = []
for threshold in threshold_grid:
    prediction = (train_oof_probability >= threshold).astype(int)
    threshold_scores.append(
        {
            "threshold": threshold,
            "precision": precision_score(y_train, prediction, zero_division=0),
            "recall": recall_score(y_train, prediction, zero_division=0),
            "f1": f1_score(y_train, prediction, zero_division=0),
        }
    )
threshold_df = pd.DataFrame(threshold_scores)
selected_threshold = float(threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"])
threshold_df.to_csv(RESULT_DIR / "threshold_analysis.csv", index=False, encoding="utf-8-sig")

test_probability = test_probabilities[best_model_name]
test_prediction = (test_probability >= selected_threshold).astype(int)
final_metrics = {
    "threshold": selected_threshold,
    **threshold_metrics(y_test, test_prediction),
    "roc_auc": roc_auc_score(y_test, test_probability),
    "pr_auc": average_precision_score(y_test, test_probability),
}
print("训练集折外最优阈值：", f"{selected_threshold:.3f}")
print(final_metrics)

# %%
# 图 8：阈值曲线。阈值越高，精确率通常越高，但召回率下降。
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(threshold_df["threshold"], threshold_df["precision"], label="精确率", color="#457B9D")
ax.plot(threshold_df["threshold"], threshold_df["recall"], label="召回率", color="#E76F51")
ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", color="#2A9D8F", linewidth=2.2)
ax.axvline(selected_threshold, linestyle="--", color="black", label=f"选择阈值 {selected_threshold:.3f}")
ax.set_xlabel("分类阈值")
ax.set_ylabel("指标")
ax.set_title("图 8  训练集折外预测的阈值权衡")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "08_threshold_tradeoff.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# 图 9：最终模型的混淆矩阵。
cm = confusion_matrix(y_test, test_prediction)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["预测未购买", "预测购买"],
    yticklabels=["实际未购买", "实际购买"],
    ax=ax,
)
ax.set_title(f"图 9  {best_model_name} 混淆矩阵（阈值 {selected_threshold:.3f}）")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "09_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()

# 保存测试集预测结果，包含会话在合并数据中的索引，方便业务复查。
test_predictions = pd.DataFrame(
    {
        "data_index": X_test.index,
        "actual": y_test.values,
        "predicted_probability": test_probability,
        "predicted_label": test_prediction,
    }
)
test_predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ## 9. 解释模型：置换重要性与 SHAP
#
# 解释分两层：
# 1. **全局**：打乱某个原始变量后，模型性能下降多少；这是“模型依赖谁”的答案。
# 2. **局部**：某个具体会话为何得到高概率；SHAP 把预测拆成各特征的正负贡献。
#
# 两者都只解释模型，不证明现实中的因果方向。

# %%
# 原始变量置换重要性。相比直接读取树模型的不纯度重要性，
# 这种方法对“类别编码展开成多列”的偏差更小，并且能直接对应业务字段。
data_permutation = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
permutation_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": data_permutation.importances_mean,
        "importance_std": data_permutation.importances_std,
    }
).sort_values("importance_mean", ascending=False)
permutation_df.to_csv(RESULT_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")

# %%
# 图 10：置换重要性前 12 名。误差线来自重复置换的波动。
top_perm = permutation_df.head(12).sort_values("importance_mean")
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.barh(top_perm["feature"], top_perm["importance_mean"], xerr=top_perm["importance_std"], color="#2A9D8F")
ax.set_xlabel("打乱该变量后 ROC-AUC 的平均下降")
ax.set_ylabel("")
ax.set_title("图 10  原始变量置换重要性（测试集）")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "10_permutation_importance.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# SHAP 的 TreeExplainer 需要预处理后的数值矩阵。
# 为了可读性和速度，只解释 500 条测试会话，不改变模型参数。
best_preprocessor = best_model.named_steps["preprocessor"]
best_estimator = best_model.named_steps["model"]


def original_feature_name(encoded_name: str) -> str:
    """把 num__、cat__ 等编码列名还原为原始业务字段。"""
    core = encoded_name.split("__", 1)[-1]
    for col in categorical_columns:
        if core == col or core.startswith(col + "_"):
            return col
    return core


if isinstance(
    best_estimator,
    (DecisionTreeClassifier, RandomForestClassifier, HistGradientBoostingClassifier),
):
    shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
    X_shap = best_preprocessor.transform(shap_sample)
    feature_names = best_preprocessor.get_feature_names_out()
    shap_plot_names = [name.split("__", 1)[-1] for name in feature_names]
    explainer = shap.TreeExplainer(best_estimator)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # 把编码后的列名还原成原始业务字段，再聚合绝对值平均，便于报告解释。
    shap_aggregated = pd.DataFrame(
        {
            "feature": [original_feature_name(name) for name in feature_names],
            "abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).groupby("feature", as_index=False)["abs_shap"].sum()
    shap_aggregated = shap_aggregated.sort_values("abs_shap", ascending=False)
    shap_aggregated.to_csv(RESULT_DIR / "shap_global_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=shap_plot_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("图 11  SHAP 全局平均绝对贡献")
    plt.xlabel("平均绝对 SHAP 值")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "11_shap_global_bar.png", dpi=180, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=shap_plot_names,
        max_display=15,
        show=False,
    )
    plt.title("图 12  SHAP 特征贡献方向与分布")
    plt.xlabel("SHAP 值")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "12_shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.show()
else:
    print("最佳模型不是树模型，跳过 TreeExplainer；仍然保留置换重要性。")
    shap_aggregated = pd.DataFrame(columns=["feature", "abs_shap"])

# %%
# 逻辑回归系数作为“方向性”的补充解释。
# 注意：系数是控制其他入模特征后的偏关联，不是因果效应；共线特征也会让系数不稳定。
logreg_best = searches["LogisticRegression"].best_estimator_
logreg_preprocessor = logreg_best.named_steps["preprocessor"]
logreg_model = logreg_best.named_steps["model"]
logreg_feature_names = logreg_preprocessor.get_feature_names_out()

coefficient_df = pd.DataFrame(
    {
        "encoded_feature": logreg_feature_names,
        "coefficient": logreg_model.coef_[0],
        "odds_ratio": np.exp(np.clip(logreg_model.coef_[0], -20, 20)),
        "original_feature": [original_feature_name(name) for name in logreg_feature_names],
    }
).sort_values("coefficient", key=np.abs, ascending=False)
coefficient_df.to_csv(RESULT_DIR / "logistic_coefficients.csv", index=False, encoding="utf-8-sig")
coefficient_df.head(15)

# %% [markdown]
# ## 10. 去掉 PageValues 的稳健性检查
#
# `PageValues` 与交易价值直接相关，而且 0 值与购买几乎呈现明显的两群结构。
# 为避免模型表现完全依赖一个含义特殊的变量，这里在保持最佳模型类型和最佳超参数不变的情况下，
# 去掉 `PageValues` 及其派生指示变量，重新训练并比较 PR-AUC。
#
# 这不是因果删除实验；它只回答“离开这个强预测变量，模型还剩多少预测能力”。

# %%
drop_for_ablation = ["PageValues", "PageValueSeen"]
X_train_ablation = X_train.drop(columns=drop_for_ablation)
X_test_ablation = X_test.drop(columns=drop_for_ablation)
ablation_numeric = [col for col in numeric_columns if col not in drop_for_ablation]
ablation_categorical = categorical_columns.copy()

# 从网格搜索结果中取出模型超参数，保留相同学习算法，只移除变量。
best_params = searches[best_model_name].best_params_
ablation_model_params = {
    key.replace("model__", ""): value for key, value in best_params.items()
}

if best_model_name == "LogisticRegression":
    ablation_estimator = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
        **ablation_model_params,
    )
elif best_model_name == "DecisionTree":
    ablation_estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, **ablation_model_params)
elif best_model_name == "RandomForest":
    ablation_estimator = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, **ablation_model_params
    )
else:
    ablation_estimator = HistGradientBoostingClassifier(
        max_iter=250,
        early_stopping=False,
        random_state=RANDOM_STATE,
        **ablation_model_params,
    )

ablation_pipeline = Pipeline(
    [
        ("preprocessor", make_preprocessor(ablation_numeric, ablation_categorical)),
        ("model", ablation_estimator),
    ]
)
ablation_pipeline.fit(X_train_ablation, y_train)
ablation_probability = ablation_pipeline.predict_proba(X_test_ablation)[:, 1]

ablation_metric = {
    "model": f"{best_model_name}_without_PageValues",
    "roc_auc": roc_auc_score(y_test, ablation_probability),
    "pr_auc": average_precision_score(y_test, ablation_probability),
}
main_metric = metric_df.loc[metric_df["model"] == best_model_name].iloc[0].to_dict()
ablation_comparison = pd.DataFrame(
    [
        {"scenario": "包含 PageValues", "roc_auc": main_metric["roc_auc"], "pr_auc": main_metric["pr_auc"]},
        {
            "scenario": "移除 PageValues",
            "roc_auc": ablation_metric["roc_auc"],
            "pr_auc": ablation_metric["pr_auc"],
        },
    ]
)
ablation_comparison.to_csv(RESULT_DIR / "pagevalues_ablation.csv", index=False, encoding="utf-8-sig")
ablation_comparison.round(4)

# %% [markdown]
# ## 11. 独立的因果分析：周末访问是否提高购买率？
#
# 这里不再使用“特征重要性=因果”的说法，而是明确写出因果问题、处理变量、结局和识别假设。
#
# - 处理变量：`Weekend=1`，周末访问；`Weekend=0`，工作日访问。
# - 结局：`Revenue=1`，完成购买。
# - 调整变量：月份、特殊节日接近度、访客类型、系统、浏览器、地区、流量类型。
# - **不调整**页面浏览、停留、退出率、PageValues：这些发生在访问过程中，可能是处理后的中介变量，调整会挡住部分效应。
# - 估计目标：周末组与工作日组的平均购买率差（ATE 方向的风险差）。
#
# 识别假设：一致性、条件可交换性、正性、SUTVA。观察数据无法验证“没有未测量混杂”，
# 所以结论应写成“在假设下估计为……”，不能写成“已经证明周末导致……”。

# %%
# 图 13：因果图。用图形明确哪些变量在治疗前调整、哪些是治疗后中介。
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

boxes = {
    "基线情境": (1.2, 4.2, "#DDEBF7"),
    "周末访问\nT": (4.7, 4.2, "#FCE4D6"),
    "购买\nY": (8.0, 4.2, "#E2F0D9"),
    "会话行为\n(浏览/时长/PageValues)": (4.7, 1.4, "#EDEDED"),
}
for label, (box_x, box_y, color) in boxes.items():
    width = 2.2 if "会话" not in label else 3.0
    box = FancyBboxPatch(
        (box_x - width / 2, box_y - 0.65),
        width,
        1.3,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="#444444",
    )
    ax.add_patch(box)
    ax.text(box_x, box_y, label, ha="center", va="center", fontsize=11)

ax.annotate("", xy=(3.85, 4.2), xytext=(2.35, 4.2), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.annotate("", xy=(6.85, 4.2), xytext=(5.85, 4.2), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.annotate("", xy=(8.0, 3.45), xytext=(1.8, 3.45), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.annotate("", xy=(5.3, 2.15), xytext=(3.4, 3.55), arrowprops=dict(arrowstyle="->", lw=1.3, linestyle="--"))
ax.annotate("", xy=(8.0, 3.45), xytext=(5.7, 2.15), arrowprops=dict(arrowstyle="->", lw=1.3, linestyle="--"))
ax.text(1.7, 4.55, "已测量并调整", fontsize=9, color="#333333")
ax.text(5.2, 2.35, "可能的中介：不纳入混杂调整", fontsize=9, color="#555555")
ax.set_title("图 13  周末访问影响购买率的简化因果图")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "13_causal_dag.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# 因果分析的预处理与预测模型分开：
# 这里只纳入访问前/身份/渠道变量，避免把访问中行为当作混杂变量。
baseline_categorical = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
causal_numeric = ["SpecialDay"]
causal_columns = baseline_categorical + causal_numeric

causal_preprocessor = make_preprocessor(causal_numeric, baseline_categorical)
X_causal = df[causal_columns].copy()
T = df["Weekend"].astype(int).to_numpy()
Y = df["Revenue"].astype(int).to_numpy()
X_causal_matrix = causal_preprocessor.fit_transform(X_causal)
causal_feature_names = causal_preprocessor.get_feature_names_out()


def fit_propensity_and_outcomes(X_design, treatment, outcome):
    """估计倾向得分、加权风险与 AIPW 平均效应。"""
    # 倾向模型：预测每个会话在周末发生的概率。
    propensity_model = LogisticRegression(max_iter=3000, solver="liblinear")
    propensity_model.fit(X_design, treatment)
    propensity = propensity_model.predict_proba(X_design)[:, 1]
    propensity = np.clip(propensity, 0.02, 0.98)

    # IPW：先把两组的样本权重按组内均值归一化，再比较加权购买风险。
    w_treated = treatment / propensity
    w_control = (1 - treatment) / (1 - propensity)
    w_treated_norm = w_treated / w_treated.mean()
    w_control_norm = w_control / w_control.mean()

    risk_weekend_ipw = np.average(outcome[treatment == 1], weights=w_treated_norm[treatment == 1])
    risk_weekday_ipw = np.average(outcome[treatment == 0], weights=w_control_norm[treatment == 0])
    ate_ipw = risk_weekend_ipw - risk_weekday_ipw

    # 结果模型按处理组分别拟合，避免强行假设两组每个特征的效应完全相同。
    outcome_model_weekend = LogisticRegression(max_iter=3000, solver="liblinear")
    outcome_model_weekday = LogisticRegression(max_iter=3000, solver="liblinear")
    outcome_model_weekend.fit(X_design[treatment == 1], outcome[treatment == 1])
    outcome_model_weekday.fit(X_design[treatment == 0], outcome[treatment == 0])

    mu1 = outcome_model_weekend.predict_proba(X_design)[:, 1]
    mu0 = outcome_model_weekday.predict_proba(X_design)[:, 1]

    # AIPW：两个结果模型给出反事实预测，倾向权重再用残差纠正模型误差。
    pseudo_outcome = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / propensity
        - (1 - treatment) * (outcome - mu0) / (1 - propensity)
    )
    ate_aipw = pseudo_outcome.mean()

    return {
        "propensity": propensity,
        "risk_weekend_ipw": risk_weekend_ipw,
        "risk_weekday_ipw": risk_weekday_ipw,
        "ate_ipw": ate_ipw,
        "ate_aipw": ate_aipw,
        "standardized_risk_weekend": mu1.mean(),
        "standardized_risk_weekday": mu0.mean(),
    }


causal_point = fit_propensity_and_outcomes(X_causal_matrix, T, Y)
naive_risk_weekend = Y[T == 1].mean()
naive_risk_weekday = Y[T == 0].mean()
naive_difference = naive_risk_weekend - naive_risk_weekday

print(f"未调整：周末购买率 {naive_risk_weekend:.4f}，工作日 {naive_risk_weekday:.4f}")
print(f"未调整风险差：{naive_difference:+.4f}")
print(f"IPW 风险差：{causal_point['ate_ipw']:+.4f}")
print(f"AIPW 风险差：{causal_point['ate_aipw']:+.4f}")

# %%
# 标准化均值差（SMD）检查：加权后各协变量在两组之间是否更平衡。
def weighted_smd(matrix, treatment, weights):
    """逐列计算处理组与对照组均值差，并除以合并标准差。"""
    diff = []
    eps = 1e-12
    for j in range(matrix.shape[1]):
        x = matrix[:, j]
        mean1 = np.average(x[treatment == 1], weights=weights[treatment == 1])
        mean0 = np.average(x[treatment == 0], weights=weights[treatment == 0])
        var1 = np.average((x[treatment == 1] - mean1) ** 2, weights=weights[treatment == 1])
        var0 = np.average((x[treatment == 0] - mean0) ** 2, weights=weights[treatment == 0])
        pooled = np.sqrt((var1 + var0) / 2)
        diff.append((mean1 - mean0) / (pooled + eps))
    return np.asarray(diff)


weights = np.where(T == 1, 1 / causal_point["propensity"], 1 / (1 - causal_point["propensity"]))
# 对逆概率权重做组内归一化，只用于平衡检查的可视化。
smd_before = weighted_smd(X_causal_matrix, T, np.ones_like(T))
smd_after = weighted_smd(X_causal_matrix, T, weights)
smd_df = pd.DataFrame(
    {
        "covariate": causal_feature_names,
        "SMD_before": smd_before,
        "SMD_after": smd_after,
        "abs_before": np.abs(smd_before),
    }
).sort_values("abs_before", ascending=False)
smd_df.to_csv(RESULT_DIR / "causal_smd_balance.csv", index=False, encoding="utf-8-sig")

# %%
# 图 14：倾向得分重叠图和 Love plot。
# 倾向得分重叠支持正性假设：两组在相似概率区间内均有样本。
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(
    causal_point["propensity"][T == 0], bins=25, alpha=0.55, label="工作日", color="#457B9D", density=True
)
axes[0].hist(
    causal_point["propensity"][T == 1], bins=25, alpha=0.55, label="周末", color="#E76F51", density=True
)
axes[0].set_xlabel("周末倾向得分")
axes[0].set_ylabel("密度")
axes[0].set_title("倾向得分重叠")
axes[0].legend()

top_smd = smd_df.head(15).sort_values("abs_before")
axes[1].scatter(top_smd["SMD_before"], top_smd["covariate"], color="#8AA1B1", label="加权前")
axes[1].scatter(top_smd["SMD_after"], top_smd["covariate"], color="#E76F51", marker="D", label="加权后")
axes[1].axvline(0.1, color="gray", linestyle="--", linewidth=1)
axes[1].axvline(-0.1, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("标准化均值差")
axes[1].set_title("协变量平衡（绝对值越小越平衡）")
axes[1].legend()

fig.suptitle("图 14  周末效应的倾向得分诊断", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "14_causal_balance.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# 图 15：效应估计对比。
# 自助法重抽样 200 次，用于近似估计不确定性；它不是“证明因果”的技术，
# 只是在既定识别假设和模型下计算抽样波动。
n_bootstrap = 200
bootstrap_ipw = []
bootstrap_aipw = []
rng = np.random.default_rng(RANDOM_STATE)

for _ in range(n_bootstrap):
    indices = rng.integers(0, len(df), size=len(df))
    # 用原始字段行索引重建这一自助样本，保证治疗与结局同步重抽样。
    sample = df.iloc[indices]
    X_boot = sample[causal_columns]
    T_boot = sample["Weekend"].astype(int).to_numpy()
    Y_boot = sample["Revenue"].astype(int).to_numpy()

    # 自助样本单独拟合编码器，避免把原样本的类别编码复制到子样本。
    boot_preprocessor = make_preprocessor(causal_numeric, baseline_categorical)
    X_boot_matrix = boot_preprocessor.fit_transform(X_boot)
    estimate = fit_propensity_and_outcomes(X_boot_matrix, T_boot, Y_boot)
    bootstrap_ipw.append(estimate["ate_ipw"])
    bootstrap_aipw.append(estimate["ate_aipw"])

bootstrap_ipw = np.asarray(bootstrap_ipw)
bootstrap_aipw = np.asarray(bootstrap_aipw)
ci_ipw = (np.quantile(bootstrap_ipw, [0.025, 0.975]))
ci_aipw = (np.quantile(bootstrap_aipw, [0.025, 0.975]))

causal_result = pd.DataFrame(
    [
        {
            "estimand": "未调整风险差",
            "estimate": naive_difference,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        },
        {
            "estimand": "IPW 风险差",
            "estimate": causal_point["ate_ipw"],
            "ci_lower": ci_ipw[0],
            "ci_upper": ci_ipw[1],
        },
        {
            "estimand": "AIPW 风险差",
            "estimate": causal_point["ate_aipw"],
            "ci_lower": ci_aipw[0],
            "ci_upper": ci_aipw[1],
        },
    ]
)
causal_result.to_csv(RESULT_DIR / "causal_estimates.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8, 4.4))
for i, row in causal_result.iterrows():
    if pd.notna(row["ci_lower"]):
        x_error = np.array(
            [[row["estimate"] - row["ci_lower"]], [row["ci_upper"] - row["estimate"]]]
        )
    else:
        x_error = None
    ax.errorbar(
        row["estimate"],
        i,
        xerr=x_error,
        fmt="o",
        capsize=5,
        color=["#8AA1B1", "#2A9D8F", "#E76F51"][i],
    )
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_yticks(range(len(causal_result)))
ax.set_yticklabels(causal_result["estimand"])
ax.set_xlabel("周末相对工作日的购买率差（百分点以小数表示）")
ax.set_title("图 15  周末购买效应的点估计与 95% 自助置信区间")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "15_causal_effect.png", dpi=180, bbox_inches="tight")
plt.show()

causal_result

# %% [markdown]
# ## 12. 保存模型、结果与可复现信息
#
# 保存模型是为了让别人能直接载入并预测；保存参数和环境版本是为了知道结果由什么条件产生。

# %%
joblib.dump(best_model, RESULT_DIR / "best_model.joblib")

final_summary = {
    "dataset_rows": int(len(df)),
    "positive_rate": float(y.mean()),
    "best_model": best_model_name,
    "best_cv_pr_auc": float(searches[best_model_name].best_score_),
    "test_pr_auc": float(final_metrics["pr_auc"]),
    "test_roc_auc": float(final_metrics["roc_auc"]),
    "selected_threshold": float(selected_threshold),
    "test_precision": float(final_metrics["precision"]),
    "test_recall": float(final_metrics["recall"]),
    "test_f1": float(final_metrics["f1"]),
}
with open(RESULT_DIR / "final_summary.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, ensure_ascii=False, indent=2)

print(json.dumps(final_summary, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 13. 结论的正确表达
#
# 预测层面：
# - 若最佳模型的 PR-AUC 明显高于随机（随机约等于购买率 0.155），模型确实识别出了购买倾向；
# - 高重要性特征只能说明模型利用了这些信息，不能说明修改该行为必然增加购买；
# - `PageValues` 可能与交易价值直接相关，因此保留无 `PageValues` 的敏感性结果，避免夸大整体性能。
#
# 因果层面：
# - 周末效应是在指定因果图和假设下的估计，不是从特征重要性直接推出来的；
# - 若置信区间跨越 0，则没有足够证据证明周末平均提高购买率；
# - 若区间不跨越 0，也只能说明“在本数据、这些可测协变量和假设下，周末与购买率存在估计差异”；
# - 真实业务应优先做随机化试点，例如随机分配同等曝光预算到周末和工作日，才能直接识别因果效应。

# %%
print("全部分析完成。")
print(f"图表目录：{FIGURE_DIR}")
print(f"结果目录：{RESULT_DIR}")
