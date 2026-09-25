#!/usr/bin/env python3
"""Finalize reproducible engineering acceptance, never a task-success gate."""

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import torch

from lunar_rover_tasks.terrain_data.batched import SceneBank
from lunar_rover_tasks.terrain_data.nasa_sfd import sample_positions
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack, file_hash


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(1)
    root = a.run_dir
    manifest = json.loads((root / "run_manifest.json").read_text())
    xml = ET.parse(a.junit)
    cases = xml.findall(".//testcase")
    tests = {"count": len(cases), "failures": len(xml.findall(".//failure")),
             "errors": len(xml.findall(".//error")), "skipped": len(xml.findall(".//skipped"))}
    assert tests["count"] > 0 and tests["failures"] == 0 and tests["errors"] == 0 and tests["skipped"] == 0
    if a.junit.resolve() != (root / "metrics/tests.xml").resolve():
        shutil.copyfile(a.junit, root / "metrics/tests.xml")
    assert len(manifest["maps"]) == 7
    gpu, query_checks, spatial_checks = {}, {}, {}
    for record in manifest["maps"]:
        path = Path(record["manifest"])
        assert file_hash(path) == record["manifest_sha256"]
        scene = ScenePack(path)
        assert scene.meta["geographic_group"] == "nasa_npb" and scene.meta["role"] == "development"
        for source in scene.meta["sources"].values():
            assert file_hash(source["path"]) == source["sha256"]
        name = record["name"]
        if name != "real_base" and not name.endswith("craters_rocks"):
            continue
        tag = "real_base" if name == "real_base" else name.split("_")[0]
        report = json.loads((root / f"metrics/gpu_{tag}.json").read_text())
        result = report["results"][0]
        assert Path(report["scene"]).resolve() == path.resolve()
        assert result["terrain_contract"]["scenes"][0]["scene_manifest_sha256"] == file_hash(path)
        assert result["device"] == "cuda" and result["num_envs"] == 32 and result["steps"] == 32
        assert result["learning_updates"] == 0 and result["parameters_unchanged"]
        gpu[name] = {k: result[k] for k in ("num_envs", "steps", "cuda_peak_allocated_mib", "map_cache_mib", "joint_four_rover_steps_per_second", "parameters_unchanged")}
        cpu, cuda = SceneBank([path]), SceneBank([path], device="cuda")
        cpu.prefetch([0])
        cuda.prefetch([0])
        rng = np.random.default_rng(23)
        xy = torch.from_numpy(rng.uniform(-101, 101, (32, 128, 2)).astype(np.float32))
        ids = torch.zeros(32, dtype=torch.long)
        c, g = cpu.query(xy, ids), cuda.query(xy.cuda(), ids.cuda())
        torch.testing.assert_close(c.features, g.features.cpu(), atol=1e-4, rtol=1e-5, equal_nan=True)
        assert torch.equal(c.height_valid, g.height_valid.cpu())
        assert torch.equal(c.metrics_valid, g.metrics_valid.cpu())
        assert torch.equal(c.error_valid, g.error_valid.cpu())
        torch.testing.assert_close(c.height_error, g.height_error.cpu(), atol=1e-4, rtol=1e-5, equal_nan=True)
        query_checks[name] = {"samples": 4096, "max_abs_feature_difference": float(torch.nan_to_num((c.features-g.features.cpu()).abs()).max()), "support_masks_identical": True}
        cpu.close()
        cuda.close()
        if name == "real_base":
            continue
        p = scene.arrays["probability_rocks"]
        assert np.isfinite(p).all() and np.all(p >= 0) and abs(p.sum()-1) < 1e-12
        assert np.all(p[~scene.arrays["probability_rocks_eligible"]] == 0)
        np.testing.assert_allclose(p, .5*(scene.arrays["probability_background"]+scene.arrays["probability_ejecta"]), atol=1e-16)
        xy = sample_positions(p, 100000, (.1, .1), np.random.default_rng(23))
        cols = np.minimum((xy[:, 0]/.1).astype(int)*3//p.shape[1], 2)
        rows = np.minimum(((p.shape[0]*.1-xy[:, 1])/.1).astype(int)*3//p.shape[0], 2)
        empirical = np.bincount(rows*3+cols, minlength=9).reshape(3, 3)/len(xy)
        r, cidx = np.indices(p.shape)
        bins = (r*3//p.shape[0])*3 + cidx*3//p.shape[1]
        expected = np.bincount(bins.ravel(), weights=p.ravel(), minlength=9).reshape(3, 3)
        delta = float(np.max(np.abs(empirical-expected)))
        assert delta < .01
        spatial_checks[name] = {"samples": len(xy), "tiles": [3, 3], "max_abs_probability_error": delta, "limit": .01}
    visuals = json.loads((root / "metrics/visualization_manifest.json").read_text())
    assert len(visuals["viewports"]) == 3
    for path, checksum in visuals["figures"].items():
        assert file_hash(root / path) == checksum
    distribution = json.loads((root / "metrics/distribution_checks.json").read_text())
    assert all(d["passed"] for d in distribution.values())
    acceptance = {
        "passed": True, "scope": "NASA SFD map/interface engineering; not vehicle safety or task success",
        "run_id": root.name, "map_count": 7, "tests": tests, "diameter_sampling": distribution,
        "gpu": gpu, "cpu_gpu_queries": query_checks, "spatial_sampling": spatial_checks,
        "figure_files": len(visuals["figures"]), "learning_updates": 0,
        "source_products_unchanged": True, "regional_microterrain_calibrated": False,
        "limitations": manifest["limitations"],
    }
    source_paths = [
        Path("source/lunar_rover_tasks/lunar_rover_tasks/terrain_data") / name
        for name in ("nasa_sfd.py", "enhancement.py", "scene_pack.py")
    ] + [Path("scripts") / name for name in (
        "build_nasa_sfd_suite.py", "visualize_nasa_sfd_suite.py", "validate_nasa_sfd_suite.py",
        "benchmark_lunar_training_maps.py",
    )] + [Path("configs/terrain/nasa_sfd_v2.yaml"), Path("configs/terrain/npb_interface_diagnostic.yaml")]
    acceptance["validated_implementation_sha256"] = {str(p): file_hash(p) for p in source_paths}
    dump(root / "metrics/engineering_acceptance.json", acceptance)
    manifest.update(environment_acceptance="passed", engineering_acceptance="metrics/engineering_acceptance.json",
                    visualization="metrics/visualization_manifest.json")
    dump(root / "run_manifest.json", manifest)
    # Preserve the existing broad terrain audit suite; publish a named sub-result.
    suite = root.parent / "_suite"
    dump(suite / "metrics/nasa_sfd_v2_acceptance.json", acceptance)
    dump(suite / "metrics/nasa_sfd_v2_summary.json", {"run_dir": str(root.resolve()), "maps": manifest["maps"], "acceptance": str((root / "metrics/engineering_acceptance.json").resolve())})
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
