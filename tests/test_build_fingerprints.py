# test_build_fingerprints.py
# -*- coding: utf-8 -*-
"""
Build F2 fingerprints into a single JSONL file (with dummy labels).
Generates synthetic rings/curtains via test_synthetic_generators.py.
"""

import json
import numpy as np
from typing import Iterable, Tuple

from f2fp.core.fingerprints import build_fingerprint
from test_synthetic_generators import make_mixed_scene


OUT_JSONL = "run_fp.jsonl"


def write_fingerprints_with_dummy_labels(
    features: Iterable[Tuple[int, object, object, object]],
    out_jsonl: str,
) -> None:
    """
    Write one JSON record per feature:
      {
        "fp_index": int,
        "feature_id": int,
        ... (pose/stats/advanced/grid/label fields created by build_fingerprint)
      }
    """
    # Open in text mode with explicit encoding to avoid hidden encoding issues
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for fp_index, (feature_id, lon, lat, depth) in enumerate(features):
            fp = build_fingerprint(np.array([[lon, lat, depth]]).T,feature_id )  # default grid 12x12x12
            rec = fp.to_record(fp_index)
            fh.write(json.dumps(rec))
            fh.write("\n")
    print("Wrote fingerprints to {0}".format(out_jsonl))


def main() -> None:
    """Build a small synthetic set and write fingerprints JSONL."""
    # 6 rings + 6 curtains (adjust as needed)
    feats = make_mixed_scene(n_rings=6, n_curtains=6, seed=42)
    write_fingerprints_with_dummy_labels(feats, OUT_JSONL)


if __name__ == "__main__":
    main()
