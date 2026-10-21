# T11持续决策与物理时间信用实现

## 结论与状态

已接通单进程与多owner的持续决策、按参考时间换算的折扣、显式GAE参数、实际底层步计费、三组短训练入口和固定初态评估。45项针对性测试通过（0失败／错误／跳过），生产CUDA K512单条5 s检查通过；三组研究学习尚未启动，原A0/A1 checkpoint和暂停状态保留。

本次遵循[专项计划](../t11_decision_learning_repair_plan.md)。工程检查证明接口与计算实现，不证明提前HOLD已经消除或集合成功率提高。隔离测试可运行诊断PPO更新，不算三组研究学习。

## 实现语义

- `LearningTimeConfig`独立描述策略周期和折扣参考周期；既有1 s重规划、0.2 s控制、安全及请求延迟保持原义。未传此配置时保留旧collector元数据与checkpoint语义。
- A0/A1策略及GRU只在决策边界运行。5 s区间把一次原动作转换成固定odometry目标，每1 s更新合法约束并续发新票据；目标不随车体重新解码，不延长旧票据。普通制动和到达不会重采样。
- 区间只产生一条原动作／概率样本，累计实际时长折扣奖励。提前终止环境冻结，保留其终端观测；其他环境完成区间。下一决策入口才reset，不跨回合累积。
- `gamma_step=gamma**(dt/discount_reference_period_s)`，lambda同理。奖励势差、宏步累积、生产和诊断GAE以及skrl配置一致；真实终端无尾值，非终端缓冲尾部保留bootstrap。
- 新checkpoint记录`learning_time`、`base_steps`和`decisions`；JSON侧车也记录底层步数。跨周期或折扣的热恢复、learner导入直接拒绝。原默认路径不增加新元数据。
- 更新日志增加按HOLD/MOVE分组的团队优势均值。这是动作与团队优势的相关性，不是逐车因果信用。

## 三组实验与预算

入口仍为`scripts/run_upstream_t11.py`，新模式`--mode decision-credit`。首轮固定A1、seed 23、四owner×四环境、CPU skrl学习、CUDA MPPI K512×2、每批5轮PPO，三组全新同初始化Actor/Critic与Adam。初始参数SHA写入各run，suite核对一致性。

| 组 | 策略周期 | gamma/lambda参考周期 | 每环境每批策略决策数 | 底层环境步上限 |
| --- | --- | --- | --- | --- |
| A | 1 s | 0.2 s | 125 | 200,000 |
| B | 5 s | 0.2 s | 25 | 200,000 |
| C | 5 s | 1 s | 25 | 200,000 |

每批名义覆盖125 s／环境，约10,000底层环境步。计费按实际推进次数，包含终止和无效学习数据消耗，排除结束环境空转；不乘以车辆数。每次采集前预留完整向量决策，余额不足就结束，5 s组允许不足400步的未用额度。提前终止可使批次数超过20，不能以20批代替实际计费。

新准入仅限这项有界研究实验，不复用旧`signal100k`转移预算，也不宣称完整系统准入。异常保存错误记录和最近完整checkpoint；存在未提交采样、采样中断导致计费不确定时，不自动重采。评估开始前保留预算票据；中断后不自动重复消费。

### 固定初态评估

- 初始化、首次达到100,000底层步的更新边界、最终检查点：四个远距离初态和四个近距离初态分别跑随机／均值策略，各最多20 s。
- 远距离恢复既有A1第61批owner 0的物理、消息、记忆和请求状态；使用当前候选网络，隐藏状态与奖励累计归零。新的折扣配置只替换奖励核算，不载入旧Actor、Critic或Adam。
- 近距离为平地7.2 m方阵，整体朝向0°/90°/180°/270°，初始前进速度0.1 m/s，使用已有停车与5 s驻留判据。
- 最终另做HOLD／持续向心／向外三分支，使用合法消息生成目标，不进入PPO。含尾值结果明确标为反事实Critic残差，不当作on-policy优势。
- 每组预留1,600＋1,600＋2,800＝6,000评估底层步，三组18,000步；实际消耗另报。评估隔离Torch RNG，不改变训练环境或隐藏状态。

报告最大车距、离散度、合法向心比例、远距离HOLD概率、连续MOVE时长、目标跳变及每秒变化次数、路径长度、执行速度、实际停车驻留、碰撞／净空和未认证刹车。不同折扣下的原始回报不直接用于跨组排名。

筛选分别检查随机和均值模式：至少3/4远距离初态净靠拢，平均最大车距缩小量比A多至少1 m／20 s，平均离散度不增加，远距离HOLD概率不高于A，近距离成功驻留数不低于A。两种模式均满足才选为候选。B、C都满足时，只有C两种模式的靠拢量都进一步增加且HOLD／驻留不退化，才优先C；否则保留B。无候选则下一步研究合法局部方向引导，本轮不追加预算。该规则不是统计显著性或完整集合验收。

## 命令

以下说明模式已执行，只打印配置，不分配环境、不启动学习：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_terrain_cuda/bin/python \
  scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id decision_credit_preview --representation A1 \
  --mode decision-credit --decision-credit-arm all \
  --budget 200000 --env-owners 4 --num-envs 16 \
  --describe-decision-credit
```

**下列为待恢复后的命令，本次未执行。** 新run ID必须未占用；A/B/C分别写入同名前缀加`_a/_b/_c`的独立run，suite只汇总，不修改原训练队列。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_terrain_cuda/bin/python \
  scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id decision_credit_seed23_200k_01 --representation A1 \
  --mode decision-credit --decision-credit-arm all \
  --budget 200000 --env-owners 4 --num-envs 16 \
  --far-checkpoint outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/checkpoints/update_000061.pt \
  --release-training-pause
```

需要在已提交边界分段运行时，选择单组`--decision-credit-arm A|B|C`并设置`--updates-this-invocation`。恢复须使用该组原run ID及`--resume-state <该组完整checkpoint>`；不接受旧A0/A1模型导入。`--rollout-steps`和`--evaluate-every`的旧模式参数不控制新实验，新模式使用上表与三个评估阶段。

## 验证与证据

验证run：`outputs/runs/upstream_scale_t11/decision_credit_implementation_20261018_01/`。

针对性回归命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_terrain_cuda/bin/python -m pytest -q \
  tests/test_upstream_decision_learning.py tests/test_upstream_decision_experiment.py \
  tests/test_upstream_ablation_learning.py tests/test_upstream_decision_credit.py \
  tests/test_upstream_process_rollout.py tests/test_upstream_signal_admission.py \
  tests/test_upstream_training_runtime.py tests/test_upstream_skrl_migration.py \
  tests/test_upstream_checkpoint.py tests/test_upstream_training_session.py \
  --junitxml=outputs/runs/upstream_scale_t11/decision_credit_implementation_20261018_01/metrics/contracts.xml
```

CPU契约测试采用K8/K16缩小规划成本，覆盖真实采样／GAE／异步终止、四owner×四环境计费与分片恢复、实际停车驻留、固定远距离初态迁移、旧准入和恢复兼容。实验循环预算测试使用可控采样／优化器替身，不冒充真实20万步学习。

另以生产CUDA配置K512执行一条5 s持续决策，核对25个底层步与直接折扣求和；该检查0优化器更新。结果与日志分别写入`metrics/cuda_smoke.json`、`logs/cuda_smoke.log`。最终检查数及校验摘要以`metrics/validation.json`为准。三组行为比较仍待恢复学习后获得。

CUDA检查的可复跑核心命令（仅打印结果，不覆盖原证据）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_terrain_cuda/bin/python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
import torch
from run_upstream_t11 import make_collector
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
torch.set_num_threads(1)
UpstreamEnvironment.cpu_workers = 2
c = make_collector('configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml',
    'A1', 23, samples=512, num_envs=1, learning_time=LearningTimeConfig(5., 1.))
batch, _, records = c.collect(1)
assert batch['duration'].tolist() == [[25]] and c.base_steps == 25
raw = [r['reward'][0] for r in records if 'execution' in r]
assert abs(sum(c.gamma_step**k*r for k, r in enumerate(raw))-batch['reward'][0, 0].item()) < 2e-5
print(c.env.cfg.planner_device, c.base_steps, c.decisions, batch['reward'])
PY
```
