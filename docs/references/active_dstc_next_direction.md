# Active-DSTC下一方向证据简报

更新时间：2026-08-20。

## 1. 触发原因

exp161表明：站点证书、候选共识和末段原语优化分别可行，但没有任何单一路线完成“发现共同平地—一致选择—无死锁进入”。共同失败点是瞬时局部感知没有获得足够的平地证据。因此下一步不再增加一般MARL网络，而研究任务特定的主动信息获取。

## 2. 推荐链路

```text
DISCOVER：分散探索尚未观测区域
→ VERIFY：验证四车容量平地而非一般地图熵
→ EXCHANGE：有界转发候选，必要时安排重连
→ COMMIT：复用exp160全签证书
→ GATHER：复用exp161已通过的原语best response
```

有限belief只保存候选和frontier摘要，不建立完整3D神经地图。首轮高层使用确定性物理评分，不学习utility。

## 3. 论文依据与项目映射

### 3.1 主动信息获取

[Kantaros et al., RSS 2019](https://roboticsproceedings.org/rss15/p62.html)把多机器人主动信息获取写为同时搜索运动空间和可达信息空间的非短视规划问题。项目中不需要照搬其完整采样规划器，但应把目标从“立即集合进展”改为“发现可形成证书的平地概率”。

[Pragr et al., RSS 2019](https://www.roboticsproceedings.org/rss15/p40.html)在自主探索中在线维护地形穿越代价模型，并利用预测方差决定继续空间探索还是降低地形模型不确定度。这支持在`DISCOVER/VERIFY`间显式切换，而不是把所有未知区域都当作同一种frontier。

### 3.2 多机器人探索分工

[Corah and Michael, RSS 2017](https://www.roboticsproceedings.org/rss13/p70.html)的DSGA用少量分布式规划轮次减少冗余信息采集，并报告2—3轮已接近串行SGA的信息目标。本项目可用相同原则分配“谁去验证哪个候选/frontier”，但共同站点最终仍由exp160的all-to-one commit决定。

[Zhou et al., IEEE T-RO 2023](https://doi.org/10.1109/TRO.2023.3236945)的RACER展示了有限异步通信下空间分解、任务分配和层级局部规划的真实去中心化探索。它支持把共同选址从0.2 s低层Actor中分离出来，但不直接提供四车平地容量证书。

### 3.3 稀缺通信与重连

[Tian et al., RSS 2024](https://www.roboticsproceedings.org/rss20/p115.html)显式安排探索、间歇数据交换和返回阶段，并对稀缺近距通信下的信息更新时间作约束。项目可迁移“主动安排交换事件”的思想：若12 m图将长期断开，探索车辆必须在commit前回到可交换候选摘要的位置。

[ROAM, IEEE T-RO 2025](https://ieeexplore.ieee.org/abstract/document/10829726)证明一跳通信上的分布式地图与规划共识可以建立严格优化接口，但完整Riemannian地图优化对当前项目过重。首轮只保留稀疏候选图与版本一致性。

### 3.4 未知粗糙地形安全

[STEP, IEEE Transactions on Field Robotics 2024](https://ieeexplore.ieee.org/abstract/document/10779483)把不确定性感知地形评价、风险度量和动力学规划组合用于未知非结构化地形。项目当前已有局部风险和47原语，下一步只需在frontier评分中加入路径风险与证书发现概率，不需要重新实现完整MPC。

## 4. 首轮冻结实验

首轮不训练策略，比较以下两个配对条件：

1. 当前瞬时局部候选；
2. 有限候选belief + 任务特定frontier探索 + 有界重连。

每个场景记录：

- 首个有效证书出现时间；
- 96 s内证书覆盖；
- Mixed/Bottleneck分层覆盖；
- frontier重复访问率；
- proposal消息量和最大年龄；
- 通信恢复后的digest收敛轮数；
- 实际路径风险和碰撞；
- 证书形成后R4是否完成集合。

准入规则为：所有分层证书覆盖至少90%，伪证书为0，split-brain为0，collision不高于2%，timeout低于10%。若确定性Active-DSTC仍无法使Bottleneck覆盖显著提高，则下一步应检查传感器范围和地形生成中是否真实存在可容纳四车的平地区域，而不是训练新的高层网络。

## 5. 明确不做

- 不构建完整共享3D神经地图；
- 不同时加入GRU、GNN和学习消息；
- 不用Oracle集合点生成frontier；
- 不把frontier规划结果作为BC动作标签；
- 不在Actor输出后叠加R4；若采用R4，它将作为独立低层控制路线替换Pure RL Actor；
- 不把Open场景成功替代Mixed/Bottleneck分层门限。
