#!/usr/bin/env python3
"""Visible, source-labelled acceptance of NASA SFD hypothesis scene packs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import binom

from lunar_rover_tasks.terrain_data.nasa_sfd import csfd, ejecta_probability
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack, file_hash


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def raster_panels(arrays, titles, root, name, *, extent, cmap="terrain", common=True, caption="", units="高程（米）"):
    fig, axes = plt.subplots(1, len(arrays), figsize=(15, 5.6), layout="constrained", squeeze=False)
    low, high = min(np.nanmin(a) for a in arrays), max(np.nanmax(a) for a in arrays)
    for ax, z, title in zip(axes[0], arrays, titles):
        im = ax.imshow(z, extent=extent, origin="upper", cmap=cmap,
                       vmin=low if common else None, vmax=high if common else None,
                       interpolation="nearest", rasterized=True)
        ax.set(title=title, xlabel="局部 X（米）", ylabel="局部 Y（米）", aspect="equal")
        if not common:
            fig.colorbar(im, ax=ax, shrink=.7, label=units)
    if common:
        fig.colorbar(im, ax=list(axes[0]), shrink=.7, label=units)
    fig.suptitle(caption, fontsize=13)
    save(fig, root, name)


def render_seed(run, seed, root):
    base = ScenePack(run / "scenes/real_base/scene.json")
    crater = ScenePack(run / f"scenes/seed{seed}_craters/scene.json")
    full = ScenePack(run / f"scenes/seed{seed}_craters_rocks/scene.json")
    scenes = [base, crater, full]
    heights = [s.arrays["features"][0] + s.meta["height_origin_m"] for s in scenes]
    titles = ["真实底图", "底图＋NASA 分布陨石坑", "底图＋陨石坑＋岩石"]
    prefix = f"seed{seed}"
    extent = [-100, 100, -100, 100]
    caption = f"NPB 200×200 米｜seed {seed}｜原生格距 5 米；处理格距 0.1 米\n坑石为工程模型生成，未按 NPB 实测微地形标定"
    raster_panels(heights, titles, root, f"{prefix}_elevation_comparison", extent=extent, caption=caption, units="源 DTM 基准高程（米）")
    layers = [full.arrays["layer_craters"], full.arrays["layer_rocks"], full.arrays["features"][0]-full.arrays["base_height"]]
    raster_panels(layers, ["坑层高程增量", "岩石层高程增量", "总高程增量"], root, f"{prefix}_height_residuals", extent=extent, cmap="RdBu_r", common=False, caption=caption)
    cat = full.meta["feature_catalog"]
    gen = cat["generation"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, kind, title in zip(axes, ("craters", "rocks"), ("陨石坑", "岩石")):
        stats = gen["stats"][kind]
        lo, hi, area = stats["diameter_min_m"], stats["diameter_max_m"], stats["area_m2"]
        d = np.geomspace(lo, hi, 150)
        ax.plot(d, csfd(kind, d)*1e6, ":", label="NASA 未截断累计公式")
        ax.plot(d[:-1], (csfd(kind, d[:-1])-csfd(kind, hi))*1e6, label="有限上限截断模型")
        survival = np.clip((csfd(kind, d[:-1])-csfd(kind, hi))/(csfd(kind, lo)-csfd(kind, hi)), 0, 1)
        n = len(cat[kind])
        band_low, band_high = binom.ppf(.025, n, survival), binom.ppf(.975, n, survival)
        ax.fill_between(d[:-1], np.maximum(band_low, .1)*1e6/area, np.maximum(band_high, .1)*1e6/area,
                        alpha=.14, color="tab:orange", label="完整目录 95% 逐点抽样区间")
        for label, items, norm in (("完整生成目录", cat[kind], area), ("目标窗口中心目录", [i for i in cat[kind] if i["centre_in_target"]], 40000)):
            diam = np.sort([i["diameter_m"] for i in items])
            ax.step(diam, np.arange(len(diam), 0, -1)*1e6/norm, where="post", label=label)
        ax.set(xscale="log", yscale="log", xlabel="直径 D（米）", ylabel="累计数量（个／平方千米）", title=title)
        ax.set_ylim(bottom=.5*1e6/max(area, 40000))
        ax.grid(alpha=.2, which="both")
        ax.legend(fontsize=9)
    fig.suptitle(f"seed {seed}：累计大小—频率分布；截断统计，不将目录计数当作可辨识地貌计数\n目标窗口是空间裁剪；岩石密度还受非均匀位置概率影响", fontsize=12)
    save(fig, root, f"{prefix}_size_frequency")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    bounds = [-105, 105, -105, 105]
    for ax, key, title in zip(axes[:3], ("background", "ejecta", "rocks"), ("背景概率", "静态坑关联概率", "50%＋50% 混合概率")):
        values = full.arrays[f"probability_{key}"] / .01
        im = ax.imshow(values, extent=bounds, origin="upper", cmap="magma", rasterized=True)
        fig.colorbar(im, ax=ax, shrink=.65, label="位置概率密度（1／平方米）")
        ax.set_title(title)
    ax = axes[3]
    ax.imshow(base.arrays["height_valid"], extent=extent, cmap="Greys", vmin=0, vmax=2, origin="upper")
    for kind, color in (("craters", "#32577a"), ("rocks", "#dd661c")):
        circles = [Circle((i["x_m"]-100, i["y_m"]-100), i["diameter_m"]/2) for i in cat[kind]]
        ax.add_collection(PatchCollection(circles, facecolor=color, edgecolor="none", alpha=.8))
    ax.set_title("坑石目录位置（符号为实际直径）")
    for ax in axes:
        ax.add_patch(Rectangle((-100, -100), 200, 200, fill=False, color="#21bf73", lw=1))
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="局部 X（米）", ylabel="局部 Y（米）", aspect="equal")
    fig.suptitle(f"seed {seed}｜绿色框为目标窗口，外围为 5 米生成缓冲带；不是实测岩石分布", fontsize=13)
    save(fig, root, f"{prefix}_spatial_probability")
    # Deterministic close-up: largest rock with a complete 12 m square context.
    candidates = [i for i in cat["rocks"] if 6 <= i["x_m"] <= 194 and 6 <= i["y_m"] <= 194]
    chosen = max(candidates, key=lambda i: i["diameter_m"])
    xc, yc = chosen["x_m"]-100, chosen["y_m"]-100
    c0, c1 = int((xc+94)/.1), int((xc+106)/.1)
    r0, r1 = int((94-yc)/.1), int((106-yc)/.1)
    xx = (np.arange(c0, c1)+.5)*.1-100
    yy = 100-(np.arange(r0, r1)+.5)*.1
    X, Y = np.meshgrid(xx, yy)
    crops = [z[r0:r1, c0:c1] for z in heights]
    local_extent = [xx[0]-.05, xx[-1]+.05, yy[-1]-.05, yy[0]+.05]
    raster_panels(crops, titles, root, f"{prefix}_local_elevation", extent=local_extent, caption=f"seed {seed}｜同一 12×12 米区域；围绕最大内部岩石，位置可复现", units="源 DTM 基准高程（米）")
    # 3D tick labels extend outside the nominal axes box; leave explicit gaps.
    fig = plt.figure(figsize=(18, 6))
    fig.subplots_adjust(left=.015, right=.93, bottom=.10, top=.80, wspace=.35)
    zmin, zmax = min(z.min() for z in crops), max(z.max() for z in crops)
    zref = float(np.floor(zmin))
    for index, (z, title) in enumerate(zip(crops, titles), 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.plot_surface(X, Y, z-zref, cmap="terrain", vmin=zmin-zref, vmax=zmax-zref, rstride=1, cstride=1, linewidth=0, rasterized=True)
        ax.set(xlabel="X（米）", ylabel="Y（米）", zlabel="相对高程（米）", title=title, zlim=(zmin-zref, zmax-zref))
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=8, pad=1)
        ax.set_box_aspect((12, 12, max(zmax-zmin, .01)))
        ax.view_init(elev=30, azim=-65)
    fig.suptitle(f"seed {seed}｜局部三维地形，XYZ 等比例，无垂直夸大；半椭球为高度场代理\n相对高程加 {zref:g} 米即源 DTM 基准高程", fontsize=13)
    save(fig, root, f"{prefix}_local_3d")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    row = int((100-yc)/.1)
    for s, title in zip(scenes, titles):
        z = s.arrays["features"][0][row, c0:c1]
        axes[0].plot(xx, z+s.meta["height_origin_m"], label=title)
        axes[1].plot(xx, z-base.arrays["base_height"][row, c0:c1], label=title)
    axes[0].set(ylabel="源基准高程（米）", title=f"seed {seed}｜穿过所选岩石中心附近的东西向剖面")
    axes[1].set(xlabel="局部 X（米）", ylabel="相对底图增量（米）")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    save(fig, root, f"{prefix}_profile")
    return {"seed": seed, "rock_id": chosen["id"], "centre_local_xy_m": [xc, yc], "extent_xy_m": local_extent, "vertical_exaggeration": 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.run_dir / "figures"
    root.mkdir(exist_ok=True)
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    viewports = []
    for seed in (23, 24, 25):
        viewports.append(render_seed(a.run_dir, seed, root))
        print(f"Rendered seed {seed}", flush=True)
    records = json.loads((a.run_dir / "metrics/map_checks.json").read_text())
    complete = [r for r in records if r["name"].endswith("craters_rocks")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    labels = ["23", "24", "25"]
    for ax, kind, title in zip(axes[0, :2], ("craters", "rocks"), ("目标窗口坑中心数量", "目标窗口岩石中心数量")):
        ax.bar(labels, [r["populations"][kind]["target_centres"] for r in complete])
        ax.set_title(title)
    axes[0, 2].bar(labels, [r["residual_rms_m"] for r in complete])
    axes[0, 2].set_title("高程增量 RMS（米）")
    for ax, field, title in zip(axes[1, :2], ("generation_coverage_fraction", "generation_overlap_fraction"), ("生成范围坑覆盖率", "生成范围坑重叠率")):
        ax.bar(labels, [r["geometry"]["craters"][field] for r in complete])
        ax.set_title(title)
    axes[1, 2].bar(labels, [r["native_scale_diagnostics"]["native_area_mean_residual_rms_m"] for r in complete])
    axes[1, 2].set_title("回到 5 米网格的增量 RMS（米）")
    for ax in axes.flat:
        ax.set_xlabel("随机种子（同一地理区域）")
    fig.suptitle("三种子验收汇总｜目录数量不等于可辨识地貌数量；不是三个独立地理样本")
    save(fig, root, "seed_summary")
    single = ejecta_probability((101, 101), (.1, .1), [{"diameter_m": 2, "x_m": 5.05, "y_m": 5.05}])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    im = axes[0].imshow(single, extent=[-5.05, 5.05, -5.05, 5.05], origin="upper", cmap="magma")
    fig.colorbar(im, ax=axes[0], label="未归一化关联权重")
    axes[0].set(xlabel="距坑中心 X（米）", ylabel="距坑中心 Y（米）", title="直径 2 米单坑：坑内抑制、坑外关联")
    radial = np.arange(51)*.1
    axes[1].plot(radial, single[50, 50:], label="物理像元中心采样")
    outer = np.linspace(1.000001, 2, 100)
    axes[1].plot(outer, np.exp(-outer/.7), "--", label="坑外理论 exp(−r/0.7)，r 为半径比")
    axes[1].set(xlabel="距坑中心（米；此例半径为 1 米）", ylabel="未归一化关联权重", title="静态径向规则；年龄衰减关闭")
    axes[1].legend(fontsize=9)
    save(fig, root, "single_crater_spatial_diagnostic")
    result = {"viewports": viewports, "figures": {str(p.relative_to(a.run_dir)): file_hash(p) for p in root.iterdir() if p.suffix in (".png", ".pdf")}}
    (a.run_dir / "metrics/visualization_manifest.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
