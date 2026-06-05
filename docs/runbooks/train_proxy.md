# Proxy 训练操作手册

## 标准命令

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/<config>.yaml \
  --output-layout run \
  --run-name <run_id> \
  --device cuda
```

CPU 只用于 smoke test，或 CUDA 不可用时的链路验证。

## 冒烟测试

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml \
  --output-layout run \
  --run-name smoke_seed23_16k_strong_lunar_crater_cpu \
  --device cpu \
  --seed 23 \
  --total-env-steps 16384 \
  --eval-num-envs 64 \
  --eval-steps 60
```

## 续训

续训必须使用明确的 `run_name`，并关闭 warm-up：

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml \
  --output-layout run \
  --run-name <retry_run_id> \
  --device cuda \
  --seed <seed> \
  --resume-checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --bc-steps 0 \
  --total-env-steps 6000000 \
  --learning-rate 0.00004 \
  --eval-num-envs 1024 \
  --eval-steps 260
```

## 规则

- 必须设置 `--run-name`。
- 不覆盖已经失败的 run；失败结果是诊断资料。
- 独立评估没有通过 strict gate 时，不报告成功。
- 强地形实验中，除非实验目标明确要求，否则不要为了通过而降低地形强度。
