# XGrammar 性能测试结果

## 1. Issue #805：`minLength`

复现脚本：[01_issue805_min_length.py](01_issue805_min_length.py)

| 单次 fill 中位数 | 128 | 129 | 200 |
|---|---:|---:|---:|
| exact `801cc1e` | 9.701 ms | 42.888 ms | 56.697 ms |
| `0.2.6rc1` 默认 | 98.751 ms | 1,707.637 ms | 4,487.605 ms |
| `0.2.6rc1` dynamic | 110.043 ms | 108.810 ms | 108.179 ms |

结论：rc1 默认模式仍存在明显的 128→129 性能断崖；dynamic 模式消除了断崖，但该压力词表下的绝对 fill 延迟仍约为 108 ms。

## 2. Issue #852：`maxLength`

复现脚本：[02_issue852_max_length.py](02_issue852_max_length.py)

| Kimi-K3，273 tokens | `maxLength=100000` compile | fill total | fill p50 | 无 `maxLength` fill total | 无 `maxLength` fill p50 |
|---|---:|---:|---:|---:|---:|
| exact `801cc1e` | 902.2 ms | 679.9 ms | 2.4578 ms | 2.6 ms | 0.0096 ms |
| `0.2.6rc1` 默认 | 871.6 ms | 797.3 ms | 2.5584 ms | 1.0 ms | 0.0037 ms |
| `0.2.6rc1` dynamic | 0.2 ms | 2.2 ms | 0.0007 ms | 2.9 ms | 0.0036 ms |

结论：只升级 rc1 无法解决 `maxLength` 的逐 token 慢路径；启用 dynamic 后，稳态 fill p50 从 2.5584 ms 降至 0.0007 ms。

## 3. `max_tokens` / `max_chars` 编译耗时

复现脚本：[03_budget_compile.py](03_budget_compile.py)

| 50k vocab 编译耗时 | 无预算 | `max_tokens=1` | `max_tokens=1,000,000` | `max_chars=1` | `max_chars=1,000,000` |
|---|---:|---:|---:|---:|---:|
| exact `801cc1e` | 0.918 ms | 0.884 ms | 0.875 ms | 127.611 ms | 125.914 ms |
| rc1 + `801cc1e` 默认 | 1.212 ms | 1.196 ms | 1.213 ms | 111.070 ms | 111.008 ms |
| rc1 + `801cc1e` dynamic | 0.152 ms | 0.153 ms | 0.150 ms | 0.146 ms | 0.147 ms |

结论：`max_tokens` 基本不增加编译耗时；`max_chars` 在默认模式下显著增加编译耗时，dynamic 将这部分工作延迟到解码阶段。

## 4. `max_chars` 边界 fill

复现脚本：[04_max_chars_boundary.py](04_max_chars_boundary.py)

| `max_chars=128`，50k vocab 单次 fill | remaining=33 | remaining=32 | remaining=16 | remaining=8 | remaining=1 | 最后 32 步总计 |
|---|---:|---:|---:|---:|---:|---:|
| exact `801cc1e` | 0.54 µs | 89.75 µs | 15.88 ms | 10.30 ms | 6.80 ms | 344.50 ms |
| rc1 + `801cc1e` 默认 | 0.50 µs | 89.08 µs | 14.03 ms | 9.84 ms | 6.84 ms | 309.42 ms |
| rc1 + `801cc1e` dynamic | 0.50 µs | 87.38 µs | 14.15 ms | 9.86 ms | 6.92 ms | 311.61 ms |

结论：当剩余字符数不大于最长 token 长度时，`max_chars` 出现明显边界慢路径；dynamic 不能消除该解码成本。

## 5. `max_chars` 词表规模扩展

复现脚本：[05_max_chars_vocab_scaling.py](05_max_chars_vocab_scaling.py)

| `max_chars` remaining=1 词表扩展 | 1k vocab | 5k vocab | 20k vocab | 50k vocab |
|---|---:|---:|---:|---:|
| exact `801cc1e` | 147 µs | 695 µs | 2.77 ms | 6.76 ms |

结论：`max_chars` 的边界 fill 延迟随词表规模近似线性增长。

## 6. `max_tokens` 预算规模扩展

复现脚本：[06_max_tokens_budget_scaling.py](06_max_tokens_budget_scaling.py)

| `max_tokens` 边界 fill | budget=1 | budget=64 | budget=1024 | budget=4096 |
|---|---:|---:|---:|---:|
| AnyText | 0.417 µs | 0.333 µs | 0.375 µs | 0.333 µs |
| AnyTokens | 0.416 µs | 0.375 µs | 0.333 µs | 0.334 µs |

结论：AnyText 和 AnyTokens 的 `max_tokens` 边界开销均保持在亚微秒级，未观察到随预算增大的性能退化。
