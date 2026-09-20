#!/usr/bin/env python3
"""Freeze a native-informed Gaussian-tail importance control without refitting."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path,
                        default=ROOT / "runs/smc-normalizer-local-atlases-s2/site0/models/all.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs/smc-normalizer-local-nativewide/site0")
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--retained-weight", type=float, default=.5)
    parser.add_argument("--extra-std-scales", type=float, nargs="+", default=[2., 4.])
    args = parser.parse_args()
    if not 0 < args.retained_weight < 1 or not args.extra_std_scales or any(not 0 < s < float("inf") for s in args.extra_std_scales):
        parser.error("positive finite scales and retained weight strictly between zero and one required")
    source = args.source_model.resolve()
    original = json.loads(source.read_text())
    count = len(original["weights"])
    assert 0 <= args.component < count
    assert original["coordinate_convention"] == "anchor-body-relative"
    assert abs(sum(original["weights"]) - 1.) < 1e-12
    for field in ["anchors", "means", "covariances"]:
        assert len(original[field]) == count
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "provenance"
    archive.mkdir()
    shutil.copy2(source, archive / "source-model.json")
    shutil.copy2(__file__, archive / "prepare_native_tail_normalizer_atlas.py")
    model = {field: copy.deepcopy(original[field]) for field in [
        "schema", "angular_length", "coordinate_convention", "shape_sha256",
        "anchors", "means", "covariances", "weights"]}
    model["weights"] = [args.retained_weight * w for w in original["weights"]]
    extra_weight = (1 - args.retained_weight) / len(args.extra_std_scales)
    for scale in args.extra_std_scales:
        model["anchors"].append(copy.deepcopy(original["anchors"][args.component]))
        model["means"].append(copy.deepcopy(original["means"][args.component]))
        model["covariances"].append([[scale * scale * v for v in row]
                                     for row in original["covariances"][args.component]])
        model["weights"].append(extra_weight)
    provenance = dict(kind="explicit native-informed tail proposal control; no physical density change",
        source_model=str(source), source_model_sha256=sha(source), retained_components=count,
        retained_total_weight=args.retained_weight, copied_source_component=args.component,
        extra_standard_deviation_scales=args.extra_std_scales, extra_component_weights=[extra_weight] * len(args.extra_std_scales),
        changes="Existing means, anchors and covariance matrices are unchanged. Existing weights share one common factor. Extra charts retain source mean/anchor and scale covariance by the square of their stated factor.",
        runtime="Keep positive uniform cube/Haar defense; evaluate the complete normalized Gaussian mixture; global covariance-scale1.",
        inference="No fit, physical energy, basin boundary, or equilibrium weight is changed. Native information is explicitly supplied.")
    model["proposal_provenance"] = provenance
    assert abs(sum(model["weights"]) - 1) < 1e-12
    for field in ["anchors", "means", "covariances"]:
        assert model[field][:count] == original[field]
    write(out / "model.json", model)
    provenance.update(output_model_sha256=sha(out / "model.json"), output_components=len(model["weights"]),
        archived_files={p.name: sha(p) for p in sorted(archive.iterdir())})
    write(out / "manifest.json", provenance)
    print(json.dumps(dict(out=str(out), components=len(model["weights"]), model_sha256=sha(out / "model.json")), indent=2))


if __name__ == "__main__":
    main()
