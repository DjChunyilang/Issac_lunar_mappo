# Windows／WSL2 部署交接：12400F、16GB、RTX 4060 Ti 8GB

更新：2026-10-26，Asia/Shanghai；文件名保留首次交接日期。本文与 `windows_wsl2_20261026_01` 新包配套，替代10月23日的部署快照。本文供目标电脑上的 Codex 接续部署。先完成环境和有界运行验收；达到本页结束条件后交付，不自动启动或恢复训练。

## 1. 用户选择与本次目标

- 目标硬件：Intel i5-12400F、16GB 系统内存、NVIDIA RTX 4060 Ti 8GB。
- 保留 Windows，优先使用 WSL2 + Ubuntu 24.04。用户不希望在非必要情况下使用双系统，也尚未要求 Windows 原生适配。
- 当前任务是部署已有 Proxy／skrl MAPPO／CUDA MPPI 项目。安装依赖、传输校验和本页有限验收属于部署工作；不增加学习预算，不启动旧实验或消融队列。
- 最小验收：源码与数据完整，Python 依赖可用，CPU／CUDA 原生内核可编译并通过针对性数值检查，四图可运行，TensorBoard 可访问且 events 更新；另用一次 16 环境采样检查这台机器的内存与吞吐。
- 结束条件：记录通过项、实际资源和未通过项，告知用户部署结果。地形失败和低成功率属于实验现象；安装错误、数据缺失、非有限数或内存不足才影响部署验收。

接手先读 [AGENTS.md](../../AGENTS.md)、[当前状态](../current_status.md)、[实施计划](../implementation_plan.md)、[架构](../architecture/planning_pipeline.md)和[接口](../interface_spec.md)。历史 `t11_training_handoff.md` 和 `setup_environment.md` 中的旧实验／全栈安装命令不作为本次默认流程。

## 2. 本机收尾与交接快照

源码基线是 `main` 的 `9c5512b0e5ccbb6ff73e7c797e70212cb1b0e8d1`，但当前实现有大量未提交和未跟踪文件。交接包含**实际工作区内容**；只克隆 Git 提交会漏掉本轮实现。

本次纳入10月24日至26日新增的学习有效性修复、百万步恢复结果、目标执行分析、MPPI车身覆盖修复、碰撞算法对照、查询优化及SAT三轮训练。最新入口为：

| 用途 | 当前文件／设置 |
| --- | --- |
| 实验配置 | `configs/experiment/upstream_1km_mppi_grid_sat_v1.yaml` |
| 训练配置 | `configs/training/upstream_1km_far_crater_reward_1m_v1.yaml` |
| 规划 | CUDA MPPI，512采样×3轮；大坑查询 `observed_grid_sat_v1` |
| 并行／学习 | 4 owner×4环境，CPU策略／PPO，每owner双线程 |
| 继承语义 | 原来的两轮配置和checkpoint保持原义，不改标为SAT三轮 |

### 2.1 最新训练已由用户停止

```text
outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/
```

10月25日20:18从零启动，10月26日14:49已按用户指令停止训练、owner及自动总结等待进程。最后已记录采样为 **911,630步**；最新完整恢复点是 **`checkpoints/update_000071.pt`，902,079步、71批／355轮PPO**。其后的9,551个已记录采样步未形成完整保存点，停止时未回传步数未知，不能把日志末点视作可恢复计数或精确总消耗。

必须同时保存同名JSON和整个 `update_000071.owners/`；`.learner.pt`只是学习器备份。源机中央和4个owner分片已按保存SHA核验。最终1500秒完整评估未完成，不能写成 `budget_complete`。最新已完成的800,020步60秒面板中，两模式均0/4窗内失败，但平均最远车距扩大2.23／0.74米；持续靠拢仍不稳定，尚无集合收敛checkpoint。

停止证据为本run `metrics/user_stop_20261026.json`，结果背景见[SAT训练记录](../experiments/t11_mppi_sat_long_training_2026-10-25.md)。TensorBoard6006仅保留本run回放；停止后events不应继续增长。**接收包不授权恢复训练，也不授权消耗原预算中未用完的部分。**

### 2.2 一并交接的前序证据

- [百万步恢复训练](../experiments/t11_crater_reward_recovery_2026-10-24.md)：新增实际采样999,812步，最终完整面板0/8成功；携带第82批完整checkpoint及前序故障run的第5批checkpoint／无效批次证据。
- [详细分析](../experiments/t11_crater_reward_detailed_analysis_2026-10-25.md)、[目标执行对照](../experiments/t11_target_execution_probe_2026-10-25.md)、[MPPI执行修复](../experiments/t11_mppi_execution_repair_2026-10-25.md)：0.5米可达目标由0/10提升至10/10，原目标仍有不可达位置；局部执行通过不代表集合成功。
- [碰撞算法对照](../experiments/t11_collision_algorithms_2026-10-25.md)、[查询优化](../experiments/t11_mppi_query_optimization_2026-10-25.md)：采用已观测栅格SAT、合并软查询和3轮迭代；圆形真值对照仍只属于oracle诊断。
- [10月23日短训](../experiments/t11_crater_reward_short_training_2026-10-23.md)：保留第4批checkpoint及结束证据，作为历史起点，不再称为最新训练。

## 3. 交接文件与数据边界

本机交付目录：

```text
outputs/runs/deployment_handoff/windows_wsl2_20261026_01/
```

将其中 `artifacts/` 的全部文件复制到目标机同一个传输目录：

| 文件 | 内容 |
| --- | --- |
| `repository.bundle` | main 分支的 Git 历史，不包含未提交内容 |
| `source_and_state.tar.gz` | 实际工作区源码、配置、文档、仓库技能；上述4组完整checkpoint；新增训练和诊断的配置、metrics、events、采样记录及准备证据；交接校验文件 |
| `runtime_data.tar.gz` | 当前四张1km地图和四张坑分档atlas，保留仓库相对路径 |
| `SHA256SUMS` | 其余5个文件（含package_summary.json）的传输校验 |
| `package_summary.json` | 随包携带的体积、文件数及本机校验结果 |
| `windows_wsl2_deployment_handoff_2026-10-23.md` | 本页的独立副本，供解包前阅读 |

归档体积、文件数和校验结果见本机交付目录 `run_manifest.json`、`metrics/verification.json`；目标机可读取随包的 `package_summary.json`。源码包内也携带交付目录的 `config/`，包含 `producer_environment.json`、`constraints.txt`、`source_manifest.json`、`runtime_manifest.json`、`runtime_files.sha256` 和 `probe_owner16.py`。

运行数据解包后约36.6 GiB：

```text
outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/scenes/
outputs/runs/local_obstacle_input/crater_classes_atlas_20261023_01/
```

四图为 `train_0`、`train_1`、`fixed_eval`、`unseen`；atlas子目录使用各scene.json的SHA256命名，保留原名。它们由Git忽略，必须随包迁移。`ScenePack`会核对scene清单中的全部文件，不能仅复制训练当下查询的数组。原始NASA下载文件和旧atlas不是本次已构建场景运行的必要输入；全历史outputs、未选中的中间checkpoint、弃用迁移准备的完整状态、Python虚拟环境、Torch编译缓存、acados编译产物不在交接包内。

本次源数据重新核对96个文件及旧归档SHA，确认地图／atlas与10月23日包相同。`runtime_data.tar.gz`仍为21,754,482,620字节，SHA256为：

```text
d0eb979a00b28b8827acc0941264070058387a6e0b78069c2d4d3b378916eca7
```

**如果已经传过旧包，可以保留同一接收目录中的这个数据文件，只替换其余5个文件，再用新 `SHA256SUMS` 校验全部文件。** 源机旧交接run保留，新run用硬链接复用相同数据归档；不要原地修改共享归档。当前源码和checkpoint必须使用新 `source_and_state.tar.gz`，不能沿用旧45MB源码包。

新包另提供 `config/included_runs.json`、`changes_since_previous.json` 和 `checkpoint_events_verification.json`，列出选中run、相对旧包的文件变化、checkpoint步数及停止边界。完整state包含采样／诊断记录，但只挑选上述4组训练checkpoint；不是全outputs备份。

建议目标SSD至少留出约100GB可用空间作为部署起点，覆盖WSL、依赖、运行数据、归档暂存和后续输出；这不是长期训练磁盘预算。地图使用只读mmap，36.6 GiB磁盘体积不等于同时占用36.6 GiB内存。

### 3.1 局域网传输，无需移动硬盘

2026-10-26源机实测地址为 `10.63.16.145`，SSH监听22端口，用户名 `yu`；地址变化时在源机用 `ip -br -4 address` 重查。Windows先运行：

```powershell
Test-NetConnection 10.63.16.145 -Port 22
```

`TcpTestSucceeded=True`后，用WinSCP选择SFTP、主机 `10.63.16.145`、端口22、用户 `yu`，使用源Linux账户的SSH凭据（不要把密码写入文档）。远端目录是：

```text
/home/yu/Academic/Luna/Issac_lunar_mappo/outputs/runs/deployment_handoff/windows_wsl2_20261026_01/artifacts/
```

下载到 `D:\lunar_transfer` 或实际可用磁盘的目录。SFTP支持断点续传，保留中断的临时文件后重试；操作依据见[WinSCP说明](https://winscp.net/eng/docs/resume)。首次连接可核对源机ED25519公钥指纹：

```text
SHA256:Diw4R5Sief5PAozZ0CLGzePU8J8HG/tPSx0oFtkWZCw
```

若目标已有WSL，也可由接手Codex在Ubuntu运行以下命令（将路径替换为实际接收目录）：

```bash
sudo apt install -y rsync openssh-client
mkdir -p /mnt/d/lunar_transfer
rsync -rtvh --partial --info=progress2 -e ssh \
  yu@10.63.16.145:/home/yu/Academic/Luna/Issac_lunar_mappo/outputs/runs/deployment_handoff/windows_wsl2_20261026_01/artifacts/ \
  /mnt/d/lunar_transfer/
cd /mnt/d/lunar_transfer
sha256sum -c SHA256SUMS
```

中断后重复同一命令。若旧数据归档已完整下载并核对上面的SHA，可给rsync加 `--exclude=runtime_data.tar.gz`，其余文件传完后仍用新SHA清单整体校验。[rsync参数说明](https://download.samba.org/pub/rsync/rsync.1)。源机保持开机、不休眠；端口检查失败时排查局域网隔离或防火墙，不必开启公网映射。

## 4. Windows 与 WSL 设置

先确认实际Windows版本、管理员权限、虚拟化是否启用和可用SSD空间。安装适用于4060 Ti的NVIDIA Windows正式驱动；若尚未安装WSL，在管理员PowerShell执行：

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl -l -v
```

按系统提示重启，确认Ubuntu的VERSION为2。Windows安装VS Code和WSL扩展，项目通过WSL窗口打开；Python解释器选WSL项目下 `.venv_terrain_cuda/bin/python`。Codex应在这个WSL工作区执行Linux命令。

在Windows的 `%UserProfile%\.wslconfig` 中合并以下设置，保留其他已有配置：

```ini
[wsl2]
memory=10GB
swap=4GB
```

先保留CPU默认分配。10GB是内存上限，不是启动即分配；4GB swap是磁盘缓冲，不能当作高速内存。没有其他WSL任务运行时执行 `wsl --shutdown`，再打开Ubuntu使设置生效。16GB主机运行训练时控制浏览器、游戏和其他开发工具的占用。

## 5. 接收源码与运行数据

项目放在Linux文件系统，例如 `~/projects/Issac_lunar_mappo_20261026`。传输归档可以先放Windows盘；持续训练不从 `/mnt/c/` 或 `/mnt/d/` 读取地图。

以下假设交接文件在Windows的 `D:\lunar_transfer`，实际路径由接手机确认后替换：

```bash
cd /mnt/d/lunar_transfer
sha256sum -c SHA256SUMS
mkdir -p ~/projects
git clone --branch main /mnt/d/lunar_transfer/repository.bundle ~/projects/Issac_lunar_mappo_20261026
cd ~/projects/Issac_lunar_mappo_20261026
tar -xzf /mnt/d/lunar_transfer/source_and_state.tar.gz
tar -xzf /mnt/d/lunar_transfer/runtime_data.tar.gz
git status --short --branch

HANDOFF=outputs/runs/deployment_handoff/windows_wsl2_20261026_01
python3 scripts/verify_upstream_transfer.py --manifest "$HANDOFF/config/source_manifest.json" --root .
sha256sum -c "$HANDOFF/config/runtime_files.sha256"
```

使用新的目标目录；已有同名工作区时先检查其修改，不覆盖。解包后出现未提交修改是本机快照的真实状态，不要 `git reset --hard`。bundle没有配置远端写权限，部署不需要推送。

scene／atlas里的原机器绝对路径有一部分是来源记录；运行数组按清单所在目录解析。保留这些JSON的原始字节，不批量替换 `/home/yu`，否则会破坏指纹。加载发生具体错误时再按对应读取代码处理。

## 6. 建立Python和原生编译环境

当前机器已验证的组合：Python3.12.3、GCC13.3、CUDA Toolkit13.0.88、Torch2.10.0+cu126。本次重新采集环境版本，详见随包 `producer_environment.json`。目标机复用Python3.12、GCC13和Toolkit13.0系列；具体补丁号与驱动写入目标机部署记录。4060 Ti的架构是sm_89；不要套用旧安装脚本对本机RTX2060的sm_75断言。

Ubuntu内：

```bash
sudo apt update
sudo apt install -y git build-essential python3.12 python3.12-venv python3.12-dev wget ca-certificates
```

从NVIDIA官方CUDA Toolkit13.0下载页选择 **Linux → x86_64 → WSL-Ubuntu**，按官方仓库步骤安装 `cuda-toolkit-13-0`。只在Windows安装GPU驱动；WSL内不安装Linux显示驱动，也不安装包含驱动的 `cuda`／`cuda-drivers` 元包。仅装PyTorch不足以提供本项目运行时需要的nvcc。

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
c++ --version
nvidia-smi
```

确认路径正确后将这两个export写入WSL的 `~/.bashrc`，并确认Codex／IDE启动的进程同样能找到CUDA_HOME。这里Torch的cu126运行库和Toolkit13.0编译工具是不同组件，本机确实使用这一组合。

在项目根目录建立新环境：

```bash
python3.12 -m venv .venv_terrain_cuda
source .venv_terrain_cuda/bin/activate
python -m pip install --upgrade pip setuptools wheel
HANDOFF=outputs/runs/deployment_handoff/windows_wsl2_20261026_01
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c "$HANDOFF/config/constraints.txt" \
  -r requirements-terrain-cuda.txt -r requirements-planner-cuda.txt
python -m pip install --no-deps -e source/lunar_rover_tasks
python -m pip check
```

`constraints.txt`来自当前已安装版本，排除了本地editable项目和当前MPPI不需要的acados。它用于限制解析版本，不要把包含本机绝对路径的pip freeze直接当作安装入口。主要版本包括skrl2.1.0、gymnasium1.2.1、numpy2.3.1、scipy1.18.1、TensorBoard2.21.0、rasterio1.4.4和osqp1.0.4。依赖下载失败应报告具体包、索引和错误，不静默升级算法库来跳过。

当前不安装Isaac Sim／Isaac Lab、Docker或acados，也不直接运行 `install_stack.sh`。`install_terrain_cuda.sh`还依赖旧 `.venv_isaaclab` 引导pip并检查sm_75，因此本次采用上面的显式命令，不修改历史脚本来迁就新机器。

## 7. 目标机最小验收

所有命令从项目根目录运行，使用上述虚拟环境。目标验收产物统一写到新的：

```bash
CHECK=outputs/runs/deployment_handoff/windows_wsl2_acceptance_20261026_01
mkdir -p "$CHECK/config" "$CHECK/logs" "$CHECK/metrics"
set -o pipefail
```

若该run已经有验收结果，选择新的编号，不覆盖。先记录 `uname -a`、`free -h`、`nvidia-smi`、`nvcc --version`、Python及pip包版本，保留失败日志。

### 7.1 导入、GPU和原生内核：0环境步／0学习更新

```bash
python - <<'PY'
import torch, skrl, rasterio, scipy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_native import library
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.get_device_capability(0) == (8, 9)
assert torch.arange(16., device='cuda').square().sum().item() == 1240.
library(False)
library(True)
print('CPU/CUDA native build and load passed')
PY

python -m pytest -q -ra \
  tests/test_upstream_native_acceleration.py::test_native_projection_and_failure_priority_match_scalar_oracle \
  tests/test_training_tensorboard.py \
  tests/test_upstream_mppi_query_optimization.py \
  tests/test_upstream_training_runtime.py::test_training_and_evaluation_collector_honor_configured_iterations \
  --junitxml="$CHECK/logs/native_tensorboard.xml" 2>&1 | tee "$CHECK/logs/native_tensorboard.log"
```

不能将CUDA检查跳过后记为通过。编译缓存由目标GPU自行生成，不复制本机缓存。

### 7.2 四图运行与TensorBoard：最多100实际底层步／0学习更新

在另一WSL终端启动或复用仅指向本次audit的服务：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/deployment_handoff/windows_wsl2_acceptance_20261026_01/audit/tensorboard \
  --host 127.0.0.1 --port 6006 --reload_interval 5
```

Windows浏览器访问 `http://localhost:6006`；若已有服务占用该端口，检查它的logdir并选择空闲端口，向用户报告实际地址。执行现有无学习审查入口：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/audit_upstream_curriculum.py \
  --config configs/experiment/upstream_1km_mppi_grid_sat_v1.yaml \
  --training-config configs/training/upstream_1km_far_crater_reward_v1.yaml \
  --output "$CHECK/audit" 2>&1 | tee "$CHECK/logs/four_scene_audit.log"
```

此处训练配置有意使用 `upstream_1km_far_crater_reward_v1.yaml` 的短窗评估预算，以匹配现有audit对预算的断言；四图、出生、奖励和决策时序保持，实验配置已切换当前SAT三轮。不要直接将此命令换成1m周期评估配置，否则旧audit的预算断言会报错。它不调用评估或学习。

该固定场景分支对四图各采一决策，最多4×25=100步，没有优化器。确认summary通过、实际步数和有限数检查正确、events生成并随采样更新。`run_upstream_t11.py --mode diagnostic`会做诊断优化更新，不能作为这里的无学习替代命令。

### 7.3 一次16环境容量检查：最多400实际底层步／0学习更新

使用交接包的 `config/probe_owner16.py`，只调用现有collector采一决策，不创建trainer／优化器；环境语义沿用当前SAT三轮实验和1m训练配置，断言实际配置为512×3，并按生产 `learning_batch_problem` 检查有效性掩码；有效的环境无效终止会单独计数。将TensorBoard切换为只加载本次 `owner16/tensorboard/` 后执行：

```bash
HANDOFF=outputs/runs/deployment_handoff/windows_wsl2_20261026_01
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout --signal=TERM --kill-after=20s 15m \
  python "$HANDOFF/config/probe_owner16.py" --output "$CHECK/owner16" \
  2>&1 | tee "$CHECK/logs/owner16.log"
```

另开终端用 `vmstat 1`、`free -h`、Windows任务管理器和 `nvidia-smi` 观察系统可用内存、持续换页和GPU占用，保存观测。首次编译／初始化时间与采样时间分开；不把一次短测称为长期稳态吞吐或完整训练内存峰值。

本次两项采样合计最多500底层步，0 PPO。若16环境OOM或持续换页，保留错误并报告该规模未通过；不要通过静默减环境数、降K512、改mmap或放宽恢复校验来声称原配置部署成功。可以交付已经完成的环境安装／单环境验收，并与用户决定后续规模或内存升级；不自动追加诊断队列。

## 8. checkpoint与后续训练边界

交接包保留最新完整checkpoint供检查与后续研究，不是要求在目标机接着跑。最新run为 `stopped_by_user`，不是 `budget_complete`；前序恢复run才是按预算结束。准备目录中的 `launch_training.py`、`finish_and_summarize.py`、`training_command.txt` 等仅为历史证据，不启动任何训练或自动等待脚本。

完整恢复还受源码、配置、地图、atlas、环境数量、owner、RNG和绝对路径身份约束。换根目录后不能假定 `--resume-state` 自动可用；也不能编辑历史JSON或跳过身份比较来绕过。若后续用户授权继续学习，再以当时实验目标确定新run／受控迁移方式和新增预算，并保留旧checkpoint原义。

任何后续学习、探针、离线拟合或恢复必须实时保存TensorBoard events；训练以实际底层环境步为横轴，评估单独计费。日志默认仅加载当次run；历史events留存。Proxy通过与Isaac／PhysX通过分别记录。

## 9. 接手Codex交付格式

完成后报告：实际Windows／WSL／Ubuntu／驱动／Python／Torch／Toolkit版本、项目路径、依赖和数据校验、内核检查、四图检查、16环境容量结果、TensorBoard地址及日志目录、消耗的诊断步数和0优化更新。将命令与证据保存在目标验收run；只记录会影响下一步的具体问题。

可直接给另一侧Codex的指令：

> 请阅读 `docs/runbooks/windows_wsl2_deployment_handoff_2026-10-23.md`，按其中步骤在这台i5-12400F／16GB／RTX4060Ti8GB Windows电脑上部署WSL2项目。使用 `windows_wsl2_20261026_01` 新包；最新SAT训练已由用户停止，完整checkpoint为902,079步的第71批。先核对现有安装与交接包，保留已有文件，完成依赖、数据、原生内核、TensorBoard及最多500步无学习验收。不要重放旧训练命令、追加训练预算或改为双系统／Windows原生。达到部署结束条件后报告结果。

## 官方参考

- [微软：安装WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- [微软：WSL内存与swap设置](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- [微软：文件系统与性能](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [NVIDIA：CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [NVIDIA：CUDA13.0下载](https://developer.nvidia.com/cuda-13-0-0-download-archive)
- [PyTorch：2.10.0历史安装命令](https://pytorch.org/get-started/previous-versions/)
