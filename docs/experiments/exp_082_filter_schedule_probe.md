# exp082：子目标过滤课程进度后验探针（reject）

## 目的

exp073 best checkpoint 的 metadata 进度为 `512`，低于 filter warmup=`4096`，正常后验评测不会执行候选替换或强制 collision/hold-zone override。本探针使用新增的评测专用 `--filter-progress-override`，隔离“提前启用 filter”是否能解决末段 timeout。

## 结果

固定 exp073 checkpoint、exp080 执行配置、seed `11023`、512 环境、320 步：

| filter step | filter applied | dmax ratio | success | collision | timeout |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `4096` | `0.4287` | `0.2062` | `0.5664` | `0` | `0.4336` |
| `8192` | `0.7933` | `0.2055` | `0.4570` | `0` | `0.5430` |

`4096` 时主要是 collision override（fraction=`0.4222`）；`8192` 的 deterministic 替换 fraction=`0.7888`。两档均远差于 filter 尚未激活的 exp080 对照。

## 结论

不在训练前或仅在评测时提前开启这个 filter。该结果只证明现有策略不能承受该执行分布变化，不能替代用相同调度重新训练的实验。产物在 `outputs/runs/exp082_filter_schedule_probe/`。
