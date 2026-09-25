#!/usr/bin/env python3
"""Render a scene beside its native source crop using identical physical axes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack, file_hash
from matplotlib import font_manager
from rasterio.windows import Window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = ScenePack(args.scene)
    meta = scene.meta
    source = Path(meta["sources"]["dtm"]["path"])
    if file_hash(source) != meta["sources"]["dtm"]["sha256"]:
        raise ValueError("Source product checksum mismatch")
    with rasterio.open(source) as ds:
        original = (
            ds.read(1, window=Window(*meta["pixel_window"]), masked=True)
            .astype(float)
            .filled(np.nan)
        )
        original = original * ds.scales[0] + ds.offsets[0]
    final = np.asarray(scene.arrays["features"][0], dtype=float) + meta["height_origin_m"]
    final = np.where(scene.arrays["height_valid"], final, np.nan)
    dx, dy = meta["extent_xy_m"]
    extent = [-dx / 2, dx / 2, -dy / 2, dy / 2]
    vmin, vmax = (
        min(np.nanmin(original), np.nanmin(final)),
        max(np.nanmax(original), np.nanmax(final)),
    )
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.5))
    fig.subplots_adjust(left=0.075, right=0.855, bottom=0.22, top=0.79, wspace=0.22)
    label = "真实基础图" if meta["classification"] == "real_base" else "真实底图＋假设增强"
    native = meta["native_spacing_xy_m"]
    processed = meta["processing_spacing_xy_m"]
    titles = [
        f"原始 DTM 裁剪｜{native[0]:g} × {native[1]:g} 米格距",
        f"环境地图包｜{processed[0]:g} × {processed[1]:g} 米处理格距",
    ]
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad("#dedede")
    for ax, z, title in zip(axes, [original, final], titles, strict=True):
        im = ax.imshow(
            z,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        h, w = z.shape
        x = np.linspace(-dx / 2 + dx / w / 2, dx / 2 - dx / w / 2, w)
        y = np.linspace(dy / 2 - dy / h / 2, -dy / 2 + dy / h / 2, h)
        levels = np.linspace(vmin, vmax, 7)[1:-1]
        lines = ax.contour(
            x,
            y,
            np.ma.masked_invalid(z),
            levels=levels,
            colors="#283b39",
            linewidths=0.65,
            alpha=0.75,
        )
        ax.clabel(lines, fmt="%.1f", fontsize=8)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("局部投影 X（米）")
        ax.set_ylabel("局部投影 Y（米）")
        ax.set_aspect("equal")
    cax = fig.add_axes([0.89, 0.235, 0.018, 0.54])
    fig.colorbar(im, cax=cax).set_label("高程（米，源 DTM 高程基准）", labelpad=12)
    fig.suptitle(f"{meta['source_id']}  高程验收图", fontsize=20, y=0.965, weight="bold")
    fig.text(
        0.5,
        0.892,
        f"{label}  ·  覆盖 {dx:g} × {dy:g} 米  ·  两图使用相同坐标范围和色标",
        ha="center",
        fontsize=12,
        color="#334155",
    )
    fig.text(
        0.075,
        0.12,
        f"处理图高程：{np.nanmin(final):.2f} 至 {np.nanmax(final):.2f} 米"
        f"；区域高差 {np.nanmax(final) - np.nanmin(final):.2f} 米。",
        fontsize=11,
    )
    fig.text(
        0.075,
        0.07,
        "源 DTM 包含上游重建/插值；加密处理网格不会增加独立观测。灰色表示无效数据。",
        fontsize=10,
        color="#475569",
    )
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "figures").mkdir()
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.run_dir / "figures" / f"elevation_comparison.{suffix}", dpi=180, facecolor="white"
        )
    plt.close(fig)
    manifest = {
        "experiment_id": "lunar_terrain_validation",
        "run_id": args.run_dir.name,
        "kind": "visual_acceptance",
        "scene": str(args.scene.resolve()),
        "scene_sha256": file_hash(args.scene),
        "source_sha256": file_hash(source),
        "generator_sha256": file_hash(Path(__file__)),
        "classification": meta["classification"],
        "extent_xy_m": meta["extent_xy_m"],
        "native_spacing_xy_m": native,
        "processing_spacing_xy_m": processed,
        "height_origin_m": meta["height_origin_m"],
        "elevation_min_m": float(np.nanmin(final)),
        "elevation_max_m": float(np.nanmax(final)),
        "valid_fraction": float(np.isfinite(final).mean()),
        "figures": ["figures/elevation_comparison.png", "figures/elevation_comparison.pdf"],
        "learning_training_started": False,
    }
    (args.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
