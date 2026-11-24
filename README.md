# 采购数仓离线出表工具

用「关账汇总片」+「仍可读的逐日底账」精确拼出一段日区间的分组报表。
**不猜明细、不重复计数、不做片间减法，并最小化付费的底账读取天数。**

```bash
python3 -m unittest discover -s tests -t .      # 41+ 项测试（含穷举对照与随机差分）
python3 examples/build_example.py /tmp/demo      # 生成示例目录
python3 -m procurement_report /tmp/demo --start 2024-01-01 --end 2024-01-16 \
    --group-by department --group-by category --pretty
```

Python 3.11，零第三方依赖。输出为 JSON；退出码 `0` 成表、`2` 不能出表、`1` 配置/读取错误。

## 输入目录

```
<data>/catalog.json
<data>/ledger/<day>.jsonl        # 底账，逐日 JSON Lines
<data>/summaries/<slice>.json    # 关账汇总片
```

`catalog.json`：

```jsonc
{
  "ledger_version": "v7",                 // 唯一固定版次
  "dimensions": {
    "order": ["department", "category"],   // 固定维度字典
    "domains": {                           // 可选：用于校验取值与维度完整性
      "department": ["north", "south"],
      "category": ["steel", "glass"]
    }
  },
  "ledger": {
    "start": "2024-01-01",                // 半开区间 [start, end)
    "end":   "2024-01-16",
    "days": [
      {"day": "2024-01-01", "readable": true,
       "status": "verified", "file": "ledger/2024-01-01.jsonl"},
      {"day": "2024-01-03", "readable": false,
       "status": "purged",   "file": "ledger/2024-01-03.jsonl"}
    ]
  },
  "summaries": [
    {"slice_id": "jan03-08-v7", "file": "summaries/jan03-08-v7.json",
     "start": "2024-01-03", "end": "2024-01-09",   // 连续整日半开区间
     "group_dims": ["department", "category"],
     "ledger_version": "v7", "complete": true,
     "digest": "sha256:..."}
  ]
}
```

底账一行一条记录，两个非空维度 + 可空整数金额（多余字段会被忽略）：

```json
{"department":"north","category":"steel","amount":125}
{"department":"south","category":"glass","amount":null}
```

维度取值必须落在目录声明的字典内；`department`/`category`/`amount` 任一缺失、
金额既非整数也非 `null`、或整天文件缺失/损坏，都会让这次查询直接报错，
**绝不会**被当成 0 或部分数据。

汇总片文件：

```json
{
  "slice_id": "jan03-08-v7",
  "start": "2024-01-03",
  "end": "2024-01-09",
  "group_dims": ["department", "category"],
  "ledger_version": "v7",
  "complete": true,
  "groups": [
    {"values": ["north", "steel"],
     "record_count": 4, "amount_nonnull_count": 4, "amount_sum": 4000}
  ],
  "digest": "sha256:..."
}
```

`digest` 是「去掉 `digest` 字段后的规范化 JSON（键排序、无多余空白、UTF-8）」
的 SHA-256。目录中的 `digest` 与文件内容必须一致。

## 关键语义

**区间**一律是整日半开区间 `[start, end)`：底账声明范围、查询范围、汇总片范围三者一致，
相邻区间可以直接首尾相接，不会差一天。

**不存在的底账不是 0。** 目录里 `readable=false`、没有目录条目、或文件丢失，
都**不能**当成「这天没有记录」。可读底账要么整天读完，要么整次查询报 `DataError`。
底账读取按天计费，且只在**最终方案**里实际读。

**汇总片实际核对。** 只有真正可能参与出表的片才会被打开核对（目录扫描与核对都不计底账
读取）。以下任一情况该片不可用，但不会中断整次查询：
文件丢失/不可读/非 JSON、与目录的区间/维度集合/版次/完整性/摘要不符、
同组重复、数值关系非法（负计数、非空数 > 记录数、非空数为 0 却金额和不为 0）。
若某片导致本来可行的方案不可行，会出现在 `rejected_slices` 里并给出原因。

**分组字典与维度完整性。** 片内出现字典外取值 → 不可用。
片的 `group_dims` 与目录声明的是同一集合即可；文件里的顺序只决定 `values` 数组顺序。

**缺组 = 无记录**，仅对目录声明 `complete=true` 的片成立；零记录的组不会输出成行。

**合法片。** 片的整个区间必须完全落在查询区间内；必须保留所有「动过的过滤维度」
（即使白名单恰好包含字典全部取值，也算动过）和所有要求的输出分组维度。
因此按 `category` 过滤时，只按 `department` 分组的片不可用。

**精确结果。** 金额为任意精度整数（可负、可超 64 位）。均值只由
`amount_sum / amount_nonnull_count` 直接得出，**绝不对各片均值再平均**：
有行但无任何非空金额 → `average_amount = null`；无任何行 → 输出 0 行（无均值）。
每行同时给出：

- `average_amount_exact`：权威值。整除时为整数，否则为最简分数串 `"n/d"`；
- `average_amount`：便于比对的 JSON 浮点数（超大数或非二进制精确商会有精度损失，
  溢出时为 `null`）。

两者都由精确的 `amount_sum` 与 `amount_nonnull_count` 现场计算，不经过任何浮点累加。

## 规划算法

设查询有 `n` 天。`dp[i]` = 覆盖 `[i, n)` 所需的最少底账天数：

- 第 `i` 天可读 → 读它，`dp[i] ≤ 1 + dp[i+1]`；
- 某片从 `i` 开始、到 `j`（不含）→ `dp[i] ≤ dp[j]`，该片整天入账。

任何合法方案都对应从左到右推进的这条链，反之亦然，所以 DP 是**完备**的：
它不会先定一个唯一优先级再报缺失，也不会把两个重叠片相减。
同一对重叠片永远不会同时入选（片不可拆分，入选即整段占用）。

`reach[i]` 记录从 `i` 能到的最远边界：若 `reach[0] == n` 则成表；否则
`reachable_boundary` 是从左端能接到的最远日期边界（空覆盖可达左端），
并说明卡住的原因：该日既无可用的合法片开始，底账也不可读；
更早开始的片无法救场（其日期已被前缀占用，片不可拆、重叠片不可相减）。

扫描目录与核对汇总片不计底账读取；规划阶段不打开任何底账；
只有最终方案里的底账日会被读取并计数到 `ledger_days_read`。

## 输出（节选）

```json
{
  "status": "ok",
  "query": {"start": "2024-01-01", "end": "2024-01-16", "filters": {}, "group_by": ["department"]},
  "ledger_days_read": 4,
  "ledger_days": ["2024-01-01", "2024-01-02", "2024-01-09", "2024-01-10"],
  "summary_slices_used": ["jan03-08-deptonly", "jan11-16-v7"],
  "coverage": [
    {"source": "ledger", "start": "2024-01-01", "end": "2024-01-03", "days": 2,
     "ledger_days": ["2024-01-01", "2024-01-02"]},
    {"source": "summary", "start": "2024-01-03", "end": "2024-01-09", "days": 6,
     "slice_id": "jan03-08-deptonly"}
  ],
  "rows": [
    {"group": {"department": "north"}, "record_count": 10,
     "amount_nonnull_count": 10, "amount_sum": 4750,
     "average_amount": 475.0, "average_amount_exact": 475}
  ],
  "row_count": 1
}
```

不能出表的判定与「最远前缀」完全独立于最小读取方案：先做可达性 DP，可达不到右端时
报告从左边接得到的最远边界及其原因（空覆盖可达左端）。因此即使某个大区间存在重叠片，
也只有在真正存在合法覆盖时才会出表；不会先定唯一优先级再判缺失。

不能出表时：

```json
{
  "status": "cannot_produce",
  "reachable_boundary": "2024-01-11",
  "last_covered_day": "2024-01-10",
  "failure_reason": "...no usable compatible summary slice begins there, and 2024-01-11 has no readable ledger (status 'purged')...",
  "rejected_slices": [{"slice_id": "jan09-16-v6", "reason": "ledger version 'v6' does not match fixed version 'v7'"}]
}
```

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `model.py` | 类型、半开区间语义、查询规范化、片兼容性 |
| `dates.py` | 严格 ISO 整日解析 |
| `digest.py` | 规范化摘要与密封 |
| `catalog.py` | 目录解析；坏片降级为不可用而非崩溃 |
| `verify.py` | 按需核对汇总片（免费），缓存结果 |
| `planner.py` | 可达边界 DP + 最少底账读取 DP + 覆盖率自检 |
| `ledger.py` | 逐日整读、过滤、精确累加与输出行构造 |
| `engine.py` | 只按方案读取并出表 |
| `cli.py` / `__main__.py` | 命令行入口 |
| `tests/` | 单元测试、穷举对照、随机差分、独立参照实现 |

## 明确不做的事

- 不用「总额 − 已知部分」猜明细；不用两个重叠汇总片互相减数。
- 不把底账缺失、不可读或未声明当成 0。
- 不对各片均值求平均；精确值不经浮点累加（`average_amount_exact` 为权威值）。
- 不为比较方案提前读取任何未被最终选中的底账。
