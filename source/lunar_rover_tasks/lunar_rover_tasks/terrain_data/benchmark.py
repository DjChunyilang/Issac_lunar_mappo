"""Geographic/provenance grouping and reproducible, training-only scale selection."""

from __future__ import annotations

import hashlib
import itertools
import json
import math

from shapely.geometry import box


def content_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_groups(regions: dict, provenance: dict) -> tuple[list[list[str]], list[dict]]:
    names = sorted(regions)
    parent = {name: name for name in names}

    def root(name):
        while parent[name] != name:
            name = parent[name]
        return name

    edges = []
    for a, b in itertools.combinations(names, 2):
        ma, mb = regions[a]["products"]["dtm"], regions[b]["products"]["dtm"]
        if ma["crs_wkt"] != mb["crs_wkt"]:
            raise ValueError("Transform footprints to one CRS before grouping")
        ga, gb = box(*ma["bounds"]), box(*mb["bounds"])
        sa, sb = provenance["regions"][a], provenance["regions"][b]
        shared_nac = sorted(set(sa["nac_pairs"]) & set(sb["nac_pairs"]))
        shared_tc = sorted(set(sa["tc_controls"]) & set(sb["tc_controls"]))
        same_product = ma["sha256"] == mb["sha256"]
        overlap = ga.intersection(gb).area
        linked = bool(overlap or shared_nac or shared_tc or same_product)
        if linked:
            parent[root(b)] = root(a)
        edges.append(dict(a=a, b=b, footprint_gap_m=ga.distance(gb),
                          overlap_m2=overlap, shared_nac=shared_nac, shared_tc=shared_tc,
                          same_product=same_product, same_group=linked))
    groups = {}
    for name in names:
        groups.setdefault(root(name), []).append(name)
    return sorted(groups.values()), edges


def training_window_scale(regions: dict, train: list[str], evidence_scale: float) -> dict:
    lengths = []
    for name in train:
        for direction in regions[name]["spatial_structure"]["directions"].values():
            if direction["censored"]:
                return dict(size_m=None, reason="training correlation is censored; full products",
                            fitted_regions=train)
            lengths.append(direction["first_sampled_e_folding_m"])
    length = max([evidence_scale, *lengths])
    return dict(size_m=2 ** math.ceil(math.log2(length)), fitted_regions=train,
                reason="dyadic envelope of training directional e-folding and literature scale; "
                       "windows are correlated scenes, never independent samples")


def build_benchmark(regions: dict, provenance: dict) -> dict:
    groups, edges = build_groups(regions, provenance)
    if len(groups) < 3:
        raise ValueError("Fewer than three groups: nested train/validation/test unavailable")
    folds = []
    # Nested leave-group-out: no arbitrary split percentage or favourable test selection.
    for test_id, test in enumerate(groups):
        for val_id, val in enumerate(groups):
            if val_id == test_id:
                continue
            train = sorted(n for i, g in enumerate(groups) if i not in (test_id, val_id) for n in g)
            folds.append(dict(id=f"test_g{test_id}_val_g{val_id}", train=train,
                              validation=val, test=test,
                              window_design=training_window_scale(
                                  regions, train, provenance["qualitative_feature_scale_m"])))
    manifest = dict(schema="lunar_geographic_benchmark_v1", groups=groups, relations=edges,
                    folds=folds, region_ids=sorted(regions), exclusions=[],
                    provenance_sha256=content_hash(provenance),
                    product_sha256={n: regions[n]["products"]["dtm"]["sha256"] for n in regions},
                    independence_claim="Geographic and listed control-source separation only; "
                                       "four groups from one selected-site product family, "
                                       "not independent lunar population or instrument samples",
                    freeze_policy="Create a new version for changes. Never fit on validation/test.")
    manifest["manifest_sha256"] = content_hash(manifest)
    return manifest


def verify_benchmark(manifest: dict):
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if content_hash(payload) != manifest["manifest_sha256"]:
        raise ValueError("Frozen benchmark has changed")
    groups = manifest["groups"]
    all_regions = set(manifest["region_ids"])
    for fold in manifest["folds"]:
        parts = [set(fold[key]) for key in ("train", "validation", "test")]
        if set.union(*parts) != all_regions or any(a & b for a, b in itertools.combinations(parts, 2)):
            raise ValueError("Invalid region split")
        for group in groups:
            if sum(bool(set(group) & part) for part in parts) != 1:
                raise ValueError("Provenance group leakage")
        if set(fold["window_design"]["fitted_regions"]) != parts[0]:
            raise ValueError("Window design uses held-out regions")


def window_catalogue(manifest: dict, regions: dict) -> list[dict]:
    """Views retain exact geometry; partial edge windows are kept and explicitly labelled."""
    verify_benchmark(manifest)
    windows = []
    for fold in manifest["folds"]:
        size = fold["window_design"]["size_m"]
        for split in ("train", "validation", "test"):
            for name in fold[split]:
                meta = regions[name]["products"]["dtm"]
                h, w = meta["shape"]
                dx, dy = meta["spacing_m"]
                windows.append(dict(fold=fold["id"], split=split, region=name,
                                    kind="full_product", pixel_window=[0, 0, w, h]))
                if size is None:
                    continue
                for scale in (size, 2 * size):
                    nw, nh = math.ceil(scale / dx), math.ceil(scale / dy)
                    if nw >= w and nh >= h:
                        continue
                    for row in range(0, h, nh):
                        for col in range(0, w, nw):
                            windows.append(dict(fold=fold["id"], split=split, region=name,
                                                kind="view", requested_extent_m=scale,
                                                pixel_window=[col, row, min(nw, w-col), min(nh, h-row)],
                                                partial_edge=col+nw > w or row+nh > h))
    return windows
