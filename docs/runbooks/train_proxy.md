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

## exp015 两阶段 SKRL 训练

当前 GPU 有其他任务时，使用等待 PID 的单条排队命令，不终止或抢占现有进程：

```bash
mkdir -p outputs/runs/exp015_skrl_medium_soft_terrain_grid/_launcher

nohup .venv_isaaclab/bin/python scripts/run_exp015_skrl_training.py \
  --config configs/experiment/exp015_skrl_weak_warmup_medium_soft.yaml \
  --device cuda \
  --wait-pid 225869 \
  --screen-timesteps 1024 \
  --formal-timesteps 4096 \
  > outputs/runs/exp015_skrl_medium_soft_terrain_grid/_launcher/train.log 2>&1 &
```

监控：

```bash
tail -f outputs/runs/exp015_skrl_medium_soft_terrain_grid/_launcher/train.log
```

runner 会先运行 `screen_seed23_2m`。只有趋势门槛、有限值、参数更新、terrain 权重更新和动作非退化检查全部通过，才会以全新随机初始化执行 `formal_seed23_8m`。
