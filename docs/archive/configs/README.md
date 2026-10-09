# 历史诊断配置

这 6 个 `exp156_timeout_diag_*` 文件原位于 `configs/experiment/`。它们的 `extends` 指向旧 run 的 `outputs/.../metrics/paired_configs/{far_bottleneck,near_open}.json`；本机未恢复这些生成配置，干净检出也没有它们。因此这些文件不是可直接运行的实验输入，2026-10-09 从正式配置目录移入归档，保留原文供追溯。文件中的相对路径仍按**原目录**解释，不能从归档目录直接运行。

若要复现实验，应先从原 run 恢复对应生成配置并核对 hash、checkpoint 和场景 manifest，再建立自包含的 `configs/experiment/` 配置。不要用猜测的覆盖值冒充原始配对输入。当前实验进度见 [docs/current_status.md](../../current_status.md)。
