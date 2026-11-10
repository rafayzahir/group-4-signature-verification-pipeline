"""Log a finished training run (history.json + metrics.json) to Comet so the graphs can be shared.

  export COMET_API_KEY=...        (never put the key in code)
  python log_to_comet.py --run-dir runs/signature/final_model --project signature-detection
"""
import argparse
import json
import os
import time

import comet_ml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="folder with history.json and metrics.json")
    p.add_argument("--project", default="signature-detection")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    hist = json.load(open(os.path.join(a.run_dir, "history.json")))
    mpath = os.path.join(a.run_dir, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}

    exp = comet_ml.Experiment(project_name=a.project)          # reads COMET_API_KEY from the environment
    exp.set_name(a.name or f"yolox-s-signature-{time.strftime('%Y%m%d-%H%M')}")
    exp.log_parameters({"model": "YOLOX-s", "monitor": hist.get("monitor"), "patience": hist.get("patience"),
                        "best_epoch": hist.get("best_epoch"), "stopped_early": hist.get("stopped_early"),
                        "conf_threshold": metrics.get("conf_threshold"), "nms_threshold": metrics.get("nms_threshold")})
    for e in sorted(hist["epochs"], key=lambda e: e["epoch"]):
        exp.log_metric("val/mAP_0.5", e["ap50"], step=e["epoch"])
        exp.log_metric("val/mAP_0.5:0.95", e["ap50_95"], step=e["epoch"])
        if e.get("train_loss") is not None:
            exp.log_metric("train/loss", e["train_loss"], step=e["epoch"])
    for split in ("val", "test"):
        if split in metrics:
            exp.log_metrics({f"{split}/{k}": v for k, v in metrics[split].items() if isinstance(v, (int, float))})
    if os.path.exists(mpath):
        exp.log_asset(mpath)
    exp.end()
    print("Logged to Comet project:", a.project)


if __name__ == "__main__":
    main()
