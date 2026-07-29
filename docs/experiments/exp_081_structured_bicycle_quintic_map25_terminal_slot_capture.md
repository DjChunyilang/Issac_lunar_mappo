# exp081：宽触发逐槽位捕获（后验 reject）

## 目的

在共同质心校正基础上，测试末段是否应把每个 rover 的子目标直接 blend 到其专属固定槽位。该设计从不使用共享几何中点，保留 per-rover 目标。

## 配置与结果

`configs/experiment/exp081_structured_bicycle_quintic_map25_terminal_slot_capture.yaml` 在 exp080 基础上启用 capture，dmax/dispersion 触发倍数 `1.75`、blend `0.65`。固定 exp073 best checkpoint、seed `11023` 后验结果为 dmax ratio/success/collision/timeout=`0.2010/0.6074/0/0.3926`，capture active fraction=`0.2621`。

## 结论

宽触发 capture 同时破坏 dmax、success 和 timeout，reject，不训练、不进入正式执行配置。产物：`outputs/runs/exp081_structured_bicycle_quintic_map25_terminal_slot_capture/counterfactual_exp073_checkpoint_eval.json`。
