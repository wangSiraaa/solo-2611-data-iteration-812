# 多孔板布局产品与读数效应校正仓

本仓库实现两个彼此独立的组件，全部使用合成测量，不作任何医学解释：

1. **设计服务**（`plate_effects/design.py`）：管理多孔板布局、不可用孔和每样品复孔约束，
   发布带完整来源的不可变布局产品。所有编辑都是事务式的，失败整体回滚。
2. **批次读数仓**（`plate_effects/warehouse.py`）：消费公开布局产品（可来自本服务，也可
   外部导入同格式产品），对每块板独立拟合题面给定的加性模型，并按**可识别性**输出
   校正值、`UNIDENTIFIABLE` 或 `MISSING`。

## 模型与约束

对每个已用孔 `(i, j)`：

```
参考孔读数 = 期望值 + b + r_i + c_j + 误差
空白孔读数 =        0 + b + r_i + c_j + 误差
样品孔读数 =    样品值 + b + r_i + c_j + 误差
```

约束：所有行效应之和为 0，所有列效应之和为 0。“存在的行/列”按板几何（至多 8×12）
计，因此参数个数恒为 `1 + (rows-1) + (cols-1)`，8×12 板为 19。

- **空白不等于没有数据**：空白孔是显式观测，期望值为 0；未导入的已用孔才是 `MISSING`。
- **不按数组顺序 zip**：读数必须携带孔坐标（映射或 `{well, value}` 记录）。
- 重复孔、板外孔、对未用/不可用孔的测量一律拒绝导入。

## 可识别性（核心）

先用已观测的对照孔（空白 + 参考）对 `b, r, c` 做**精确有理**最小二乘，再判断唯一性：

- 参数是否唯一：约束后法方程的秩是否为满秩。
- 样品孔偏置 `b + r_i + c_j` 是否唯一：该孔特征向量是否与法方程的**精确零空间**正交。
  与零空间有非零内积的孔标记 `UNIDENTIFIABLE`，并给出证明方向（加上该方向后所有对照
  残差不变、该孔偏置恰好改变 1）。
- 只有可识别孔给 `y - (b+r_i+c_j)`；样品统计仅在**请求的全部复孔**都已观测且可识别时
  给算术平均与极差，否则 `INCOMPLETE` 并逐孔列出原因，不静默丢孔。

所有判定使用 `fractions.Fraction` 精确消元，不设秩容差；同时用 NumPy SVD 计算数值秩
与明确容差 `1e-9` 作独立交叉核对，结果写入报告。

## 目录

```
plate_effects/
  rational.py    # 有理数解析、精确 RREF/秩/零空间/线性求解
  design.py      # 事务式设计服务、布局校验、发布产品与摘要
  warehouse.py   # 读数导入、精确最小二乘、可识别性、样品统计、解释报告、缓存
  scenarios.py   # 6 个合成场景（已知 b/r/c）
  cli.py         # 演示入口
tests/
  test_plate_effects.py
  test_scenarios.py
```

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pytest
.venv/bin/python -m plate_effects.cli demo results/   # 生成 results/*.json
.venv/bin/python -m pytest -q                         # 48 个测试
```

## 场景覆盖

| 场景 | 关键点 |
|------|--------|
| `complete_controls` | 全对照，参数唯一，可识别孔校正值精确等于生成值 |
| `disconnected_controls` | 对照图断开，参数不唯一；部分孔偏置仍唯一，其余 `UNIDENTIFIABLE` |
| `unique_bias_nonunique_params` | b/r/c 不唯一但 B2 偏置唯一，可校正 |
| `missing_replicate` | 缺测复孔 → `MISSING` + 样品 `INCOMPLETE`（列出原因） |
| `role_correction` | 更正参考值/孔角色生成新版本并真实重拟合；旧 run 绑定旧布局 |
| `blocked_wells` | 不可用孔发布为 unused/blocked，拒绝测量且不出现在可用孔查询中 |

## 缓存策略

- 布局级结构（行/列集合、参数坐标）按 `layout.digest` 缓存。
- 对照法方程的精确 RREF 与行变换矩阵按 `(layout.digest, 已观测对照孔集合)` 缓存；
  不同的缺测对照会产生不同缓存键，不会误用旧分解。
- 每板上限 30，每板至多 96 孔；每板独立拟合，无跨板随机效应。
