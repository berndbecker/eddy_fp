from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import typer

from f2fp.core.synthetic import SynthConfig, ring_points, curtain_points, mix_with_noise
from f2fp.core.fingerprints import build_fingerprint, to_dict
from f2fp.core.cluster import _stack_metrics, embed_features, cluster_labels
from f2fp.io.rw import write_jsonl

app = typer.Typer(no_args_is_help=True)

@app.command()
def synth(out_jsonl: Path, center_lon: float = -30.0, center_lat: float = 15.0):
    cfg = SynthConfig(center_lon=center_lon, center_lat=center_lat)
    lon_r, lat_r = ring_points(cfg)
    lon_c, lat_c = curtain_points(cfg)
    lon_m, lat_m = mix_with_noise(np.concatenate([lon_r, lon_c]),
                                  np.concatenate([lat_r, lat_c]), cfg)
    X = np.column_stack([(lon_m-center_lon)/5.0,
                         (lat_m-center_lat)/5.0,
                         np.zeros_like(lon_m)])
    fp = build_fingerprint(X, label="synthetic_mixed")
    write_jsonl([to_dict(fp)], out_jsonl)
    typer.echo(f"Wrote {out_jsonl}")

@app.command()
def cluster(in_jsonl: Path, out_json: Path):
    fps = [json.loads(line) for line in Path(in_jsonl).read_text().splitlines() if line.strip()]
    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = _stack_metrics(fps, keys)
    E = embed_features(M, n_components=2, random_state=42)
    labels = cluster_labels(M, min_cluster_size=5, random_state=42)
    out = dict(embedding=E.tolist(), labels=labels.tolist())
    out_json.write_text(json.dumps(out))
    typer.echo(f"Wrote {out_json}")

if __name__ == "__main__":
    app()
