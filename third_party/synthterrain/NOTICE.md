# NASA synthterrain 适配说明

来源：NASA Ames Research Center，https://github.com/NeoGeographyToolkit/synthterrain

固定提交：`e10ba36e9aa5b923bc7e6746ef089859af9ccaf8`。

Copyright © 2024–2025, United States Government, as represented by the
Administrator of the National Aeronautics and Space Administration. All rights reserved.

上游采用 Apache License 2.0，完整许可证见同目录 LICENSE。

本项目的 `terrain_data/nasa_sfd.py` 适配了 `crater/functions.py`、
`rock/functions.py` 中的 VIPER 累计大小—频率公式，以及 `rock/__init__.py`
中的 OpenSimplex 背景和陨石坑关联概率规则。不是完整上游生成器复现。

修改：使用统一 SeedSequence 子流及显式有限区间逆变换；仅开放 <=80 米
坑分支；按有效中心面积计数；遮罩后分别归一化；正确的像元内坐标采样；
在真实物理像元中心计算径向衰减（不沿用上游 linspace/整像元中心定位）；
关闭年龄衰减，按稳定目录顺序叠加坑外概率并覆盖坑内概率；退化概率场
提供显式回退；保留本项目原有坑石形状。增强为工程假设，不是 NPB 观测。

上游代码称分布来自 VIPER-MSE-SPEC-001 (2021-09-16)，并注明规范内
缺少进一步引用；本实现不补写不存在的区域标定证据。
