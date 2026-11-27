# 数据说明

本项目使用 **UCI Online Shoppers Purchasing Intention Dataset**。

- 数据集主页：https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- DOI：https://doi.org/10.24432/C5F88Q
- 许可：CC BY 4.0
- 规模：12,330 条会话，17 个输入字段，1 个目标字段 `Revenue`

## 下载

从 UCI 页面下载 `online_shoppers_intention.csv`，放到本目录：

```text
data/online_shoppers_intention.csv
```

仓库不直接分发原始 CSV，避免重复分发数据；请以 UCI 官方页面为准。

## 字段

| 字段 | 说明 |
|---|---|
| Administrative / Administrative_Duration | 管理类页面数量与停留时长 |
| Informational / Informational_Duration | 信息类页面数量与停留时长 |
| ProductRelated / ProductRelated_Duration | 商品类页面数量与停留时长 |
| BounceRates / ExitRates | 跳出率与退出率 |
| PageValues | 页面价值指标 |
| SpecialDay | 距离特殊日期的接近程度 |
| Month | 访问月份 |
| OperatingSystems / Browser / Region / TrafficType | 设备、浏览器、地区、渠道类别 |
| VisitorType | 新访客 / 回访者 / 其他 |
| Weekend | 是否周末 |
| Revenue | 是否完成购买（目标变量） |

字段含义以 UCI 官方说明为准。
