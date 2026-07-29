# exp083：严格轻量逐槽位捕获（后验 reject）

## 目的

为避免 exp081 的宽触发干扰，exp083 仅在 dmax 和 dispersion 都已通过真实门时捕获，并将 blend 降至 `0.25`。配置：`configs/experiment/exp083_structured_bicycle_quintic_map25_strict_light_slot_capture.yaml`。

## 结果

固定 exp073 best checkpoint、seed `11023`、512 环境、320 步后验评测为 dmax ratio/success/collision/timeout=`0.1995/0.6191/0/0.3809`；capture active fraction=`0.1013`。虽然 dmax 仍通过，但 success 低于不启用 capture 的 exp080（`0.6250`），timeout 也更高。

## 结论

逐槽位捕获在宽触发和严格轻量两种设置下都没有收益，保持默认关闭。产物：`outputs/runs/exp083_structured_bicycle_quintic_map25_strict_light_slot_capture/counterfactual_exp073_checkpoint_eval.json`。
