#!/usr/bin/env python
"""
Train, evaluate and export a YOLOX-s signature detector (Apache 2.0, no Ultralytics).

Steps: set up YOLOX -> get data -> add negative pages -> train with early stopping
       -> tune thresholds on validation -> evaluate (val + test) -> export
       best.pt / signature.onnx / history.json / onnx_infer.py / metrics.json
       -> optional Comet logging and Hugging Face upload.

Examples:
  python train.py --quick-test
  python train.py --epochs 30 --batch-size 16 --comet --push-to-hub <user>/<repo>

Environment variables (never put keys in code):
  HF_TOKEN       Hugging Face token (dataset download; WRITE access if --push-to-hub is used)
  COMET_API_KEY  Comet key (only with --comet)

Needs a CUDA GPU for training. Files next to this script: es_wrapper.py, onnx_infer.py.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_NAME = "yolox_s_signature"
CLASSES = ["signature"]
WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"

EXP_TEMPLATE = """
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = __NC__
        self.depth, self.width = 0.33, 0.50            # YOLOX-s size
        self.data_dir = "__DATA__"
        self.train_ann = "__TRAIN_ANN__"
        self.val_ann = "__VAL_ANN__"
        self.input_size = (__SIZE__, __SIZE__)
        self.test_size = (__SIZE__, __SIZE__)
        self.max_epoch = __EPOCHS__
        self.warmup_epochs = 1
        self.no_aug_epochs = __NOAUG__
        self.eval_interval = __EVAL__
        self.print_interval = 200
        self.data_num_workers = 4
        self.save_history_ckpt = False                 # keep only last + best checkpoints
        self.exp_name = "__NAME__"
"""


# ----------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLOX-s signature detector (no Ultralytics)")
    p.add_argument("--out-dir", default="runs/signature", help="where final_model/ and logs are written")
    p.add_argument("--yolox-dir", default="YOLOX", help="YOLOX checkout (cloned here if missing)")
    p.add_argument("--data-dir", default="data/signature_coco", help="COCO-format dataset folder")
    p.add_argument("--hf-dataset", default="AimaArshad/signature-detection-yolox", help="used if data-dir is empty")
    p.add_argument("--epochs", type=int, default=30, help="maximum epochs (early stopping may end sooner)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--monitor", choices=["ap50", "ap50_95"], default="ap50", help="metric for early stopping")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--neg-ratio", type=float, default=0.10, help="extra signature-free pages (0 = none)")
    p.add_argument("--neg-max", type=int, default=2000)
    p.add_argument("--precision-target", type=float, default=0.95)
    p.add_argument("--quick-test", action="store_true", help="2 epochs on a small subset, to check the pipeline")
    p.add_argument("--skip-train", action="store_true", help="only evaluate/export an existing run")
    p.add_argument("--comet", action="store_true", help="log metrics to Comet (needs COMET_API_KEY)")
    p.add_argument("--comet-project", default="signature-detection")
    p.add_argument("--push-to-hub", default=None, metavar="USER/REPO", help="upload final_model/ to Hugging Face")
    p.add_argument("--public", action="store_true", help="make the Hugging Face repo public (default private)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r


# ----------------------------------------------------------------------------------------------
# Setup: YOLOX, pretrained weights, dataset
# ----------------------------------------------------------------------------------------------
def setup_yolox(yolox_dir):
    if not (yolox_dir / "tools" / "train.py").exists():
        sh(["git", "clone", "-q", "https://github.com/Megvii-BaseDetection/YOLOX.git", str(yolox_dir)])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=yolox_dir)
    pat = re.compile(r"\bnp\.(int|float|bool)\b")           # old YOLOX code uses names new NumPy removed
    for f in yolox_dir.rglob("*.py"):
        txt = f.read_text(encoding="utf-8")
        new = pat.sub(r"\1", txt)
        if new != txt:
            f.write_text(new, encoding="utf-8")
    weights = yolox_dir / "yolox_s.pth"
    if not weights.exists() or weights.stat().st_size < 10e6:
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    if weights.stat().st_size < 10e6:
        raise RuntimeError("yolox_s.pth download failed (file too small). Is the internet on?")
    sys.path.insert(0, str(yolox_dir))
    os.environ["PYTHONPATH"] = str(yolox_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"    # lets old YOLOX code load checkpoints
    return weights


def get_data(data_dir, hf_dataset):
    if (data_dir / "annotations" / "instances_train2017.json").exists():
        print("Dataset found at", data_dir)
        return
    print("Downloading dataset from Hugging Face:", hf_dataset)
    from huggingface_hub import hf_hub_download
    script = hf_hub_download(hf_dataset, "5_download_and_extract.py", repo_type="dataset")
    subprocess.run([sys.executable, script, "--repo", hf_dataset, "--out", str(data_dir)], check=True)


def make_subset(src, dst, n, seed=42):
    j = json.load(open(src))
    imgs = random.Random(seed).sample(j["images"], min(n, len(j["images"])))
    ids = {i["id"] for i in imgs}
    j["images"] = imgs
    j["annotations"] = [a for a in j["annotations"] if a["image_id"] in ids]
    json.dump(j, open(dst, "w"))


# ----------------------------------------------------------------------------------------------
# Negative (signature-free) pages: erase the signature from some training pages
# ----------------------------------------------------------------------------------------------
def erase_boxes(img, boxes):
    """Paint each signature box with the colour of the paper around it."""
    h, w = img.shape[:2]
    for x, y, bw, bh in boxes:
        pad = max(6, int(0.03 * max(bw, bh)))
        x1, y1 = max(0, int(x) - pad), max(0, int(y) - pad)
        x2, y2 = min(w, int(x + bw) + pad), min(h, int(y + bh) + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        rx1, ry1, rx2, ry2 = max(0, x1 - 20), max(0, y1 - 20), min(w, x2 + 20), min(h, y2 + 20)
        ring = img[ry1:ry2, rx1:rx2].copy()
        keep = np.ones(ring.shape[:2], bool)
        keep[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
        fill = np.median(ring[keep], axis=0) if keep.any() else np.array([255, 255, 255])
        img[y1:y2, x1:x2] = fill.astype(np.uint8)
    return img


def make_negatives(ann_dir, src_name, dst_name, img_dir, ratio, max_new, seed=42):
    j = json.load(open(ann_dir / src_name))
    boxes = defaultdict(list)
    for a in j["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])
    with_sig = [im for im in j["images"] if im["id"] in boxes]
    n_new = min(int(len(j["images"]) * ratio), max_new, len(with_sig))
    chosen = random.Random(seed).sample(with_sig, n_new)
    next_id = max(im["id"] for im in j["images"]) + 1
    new_imgs = []
    for k, im in enumerate(chosen):
        name = f"neg_{im['id']}.jpg"
        out = img_dir / name
        if not out.exists():
            img = cv2.imread(str(img_dir / im["file_name"]))
            if img is None:
                continue
            cv2.imwrite(str(out), erase_boxes(img, boxes[im["id"]]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        new_imgs.append({"id": next_id, "file_name": name, "width": im["width"], "height": im["height"]})
        next_id += 1
        if (k + 1) % 500 == 0:
            print(f"  negative pages: {k + 1}/{n_new}")
    j["images"] = j["images"] + new_imgs                    # new images have NO annotations = negatives
    json.dump(j, open(ann_dir / dst_name, "w"))
    return len(new_imgs)


# ----------------------------------------------------------------------------------------------
# Training (YOLOX + early stopping through es_wrapper.py)
# ----------------------------------------------------------------------------------------------
def train(args, yolox_dir, exp_file, weights, yolox_out):
    hist, last = yolox_out / "history.json", yolox_out / "last_epoch_ckpt.pth"
    if hist.exists() and json.load(open(hist)).get("finished"):
        print("Training already finished earlier. Skipping (delete", yolox_out, "to retrain).")
        return
    yolox_out.mkdir(parents=True, exist_ok=True)
    if not last.exists() and hist.exists():
        hist.unlink()                                       # fresh start -> no stale history
    env = dict(os.environ, ES_MONITOR=args.monitor, ES_PATIENCE=str(args.patience),
               ES_MIN_EPOCHS=str(args.min_epochs), ES_MIN_DELTA=str(args.min_delta),
               ES_HISTORY=str(hist), ES_YOLOX_DIR=str(yolox_dir))
    cmd = [sys.executable, "-u", str(HERE / "es_wrapper.py"), "-f", str(exp_file),
           "-d", "1", "-b", str(args.batch_size), "--fp16", "-o"]
    if last.exists():
        print("Checkpoint found -> RESUMING training")
        cmd += ["--resume", "-c", str(last)]
    else:
        print("Starting from pretrained yolox_s.pth")
        cmd += ["-c", str(weights)]
    keep = re.compile(r"EarlyStopping|epoch: \d+/\d+, iter|Traceback|Error|out of memory|Start training|Training of experiment", re.I)
    log = open(Path(args.out_dir) / "train_log.txt", "a")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=yolox_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        log.write(line)
        if keep.search(line):
            print(line.strip()[:220], flush=True)
    p.wait()
    log.close()
    print(f"Training process ended (exit code {p.returncode}) after {(time.time() - t0) / 60:.1f} min")
    # YOLOX catches its own errors and exits with code 0, so judge by history.json instead
    if not (hist.exists() and json.load(open(hist)).get("finished")):
        raise RuntimeError(f"Training did not finish. See {Path(args.out_dir) / 'train_log.txt'}")


# ----------------------------------------------------------------------------------------------
# Inference helpers + evaluation
# ----------------------------------------------------------------------------------------------
class Detector:
    def __init__(self, exp_file, ckpt_path):
        import torch
        from yolox.data.data_augment import ValTransform
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        self.torch, self.postprocess = torch, postprocess
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exp = get_exp(str(exp_file), None)
        self.model = self.exp.get_model()
        self.ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(self.ckpt["model"])
        self.model = self.model.to(self.device).eval()
        self.transform = ValTransform(legacy=False)

    def raw(self, img):
        t = self.torch
        ratio = min(self.exp.test_size[0] / img.shape[0], self.exp.test_size[1] / img.shape[1])
        x, _ = self.transform(img, None, self.exp.test_size)
        with t.no_grad():
            out = self.model(t.from_numpy(x).unsqueeze(0).float().to(self.device)).cpu()
        return out, ratio

    def decode(self, raw, ratio, conf, nms):
        out = self.postprocess(raw.clone(), self.exp.num_classes, conf, nms)[0]   # clone: postprocess edits its input
        if out is None:
            return []
        return [[float(v) for v in (*(b[:4] / ratio), b[4] * b[5])] for b in out.numpy()]

    def predict(self, img, conf, nms):
        raw, ratio = self.raw(img)
        return self.decode(raw, ratio, conf, nms)


def load_split(data_dir, split, limit=None):
    folder = f"{split}2017"
    data = json.load(open(data_dir / "annotations" / f"instances_{folder}.json"))
    gts = defaultdict(list)
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        gts[a["image_id"]].append([x, y, x + w, y + h])
    images = data["images"]
    if limit and limit < len(images):
        images = random.Random(0).sample(images, limit)
    cat_id = sorted(c["id"] for c in data["categories"])[0]
    return images, gts, data_dir / folder, cat_id


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def count_prf(images, gts, preds, thr, iou_thr=0.5):
    """Greedy matching by confidence: a box is a TP if IoU >= iou_thr with a not-yet-used real box."""
    tp = fp = fn = 0
    for im in images:
        if im["id"] not in preds:
            continue
        g, used = gts[im["id"]], set()
        for pb in sorted([p for p in preds[im["id"]] if p[4] >= thr], key=lambda p: -p[4]):
            best, bj = 0.0, -1
            for j, gb in enumerate(g):
                if j in used:
                    continue
                v = iou(pb, gb)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                tp += 1
                used.add(bj)
            else:
                fp += 1
        fn += len(g) - len(used)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(tp=tp, fp=fp, fn=fn, P=P, R=R, F1=F)


def tune_thresholds(det, data_dir, limit, precision_target):
    """Pick the confidence and NMS thresholds on the VALIDATION set only."""
    images, gts, img_dir, _ = load_split(data_dir, "val", limit)
    raws = {}
    for k, im in enumerate(images):
        img = cv2.imread(str(img_dir / im["file_name"]))
        if img is not None:
            raws[im["id"]] = det.raw(img)
        if (k + 1) % 500 == 0:
            print(f"  tuning inference: {k + 1}/{len(images)}")
    decode_all = lambda conf, nms: {i: det.decode(r, ratio, conf, nms) for i, (r, ratio) in raws.items()}

    preds = decode_all(0.01, 0.45)
    rows = [(round(float(t), 2), count_prf(images, gts, preds, float(t))) for t in np.arange(0.05, 0.96, 0.05)]
    ok = [(t, r) for t, r in rows if r["P"] >= precision_target]
    pool = ok if ok else rows
    best_conf = float(max(pool, key=lambda tr: (tr[1]["F1"], -tr[0]))[0])
    if not ok:
        print(f"WARNING: no confidence threshold reaches precision {precision_target}; using best F1.")

    nms_rows = []
    for n in (0.30, 0.40, 0.45, 0.50, 0.60, 0.65):
        nms_rows.append((n, count_prf(images, gts, decode_all(best_conf, n), best_conf)))
    best_nms = float(max(nms_rows, key=lambda nr: (nr[1]["F1"], -nr[0]))[0])
    table = lambda rs: [{"value": v, **{k: round(float(r[k]), 4) for k in ("P", "R", "F1")}} for v, r in rs]
    print(f"Chosen thresholds: confidence {best_conf}, NMS {best_nms}")
    return best_conf, best_nms, {"confidence_sweep": table(rows), "nms_sweep": table(nms_rows)}


def coco_scores(data_dir, split, images, preds, cat_id):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dets = [{"image_id": i, "category_id": cat_id, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(b[4])}
            for i, bs in preds.items() for b in bs]
    if not dets:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "mAP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(data_dir / "annotations" / f"instances_{split}2017.json"))
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = [im["id"] for im in images]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"mAP50": float(s[1]), "mAP50_95": float(s[0]), "mAP75": float(s[2])}


def evaluate_split(det, data_dir, split, limit, conf, nms):
    images, gts, img_dir, cat_id = load_split(data_dir, split, limit)
    preds = {}
    for k, im in enumerate(images):
        img = cv2.imread(str(img_dir / im["file_name"]))
        preds[im["id"]] = det.predict(img, 0.001, nms) if img is not None else []
        if (k + 1) % 500 == 0:
            print(f"  {split} inference: {k + 1}/{len(images)}")
    res = coco_scores(data_dir, split, images, preds, cat_id)
    r = count_prf(images, gts, preds, conf)
    res.update(precision=float(r["P"]), recall=float(r["R"]), f1=float(r["F1"]),
               tp=int(r["tp"]), fp=int(r["fp"]), fn=int(r["fn"]), images=len(images))
    return res


# ----------------------------------------------------------------------------------------------
# Export: best.pt, signature.onnx, onnx_infer.py, history.json, metrics.json, config, README
# ----------------------------------------------------------------------------------------------
def export_onnx(det, out_path, size):
    import copy
    torch = det.torch
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module
    m = copy.deepcopy(det.model).cpu()                      # export a COPY so the GPU model stays untouched
    m = replace_module(m, torch.nn.SiLU, SiLU).eval()
    m.head.decode_in_inference = True                       # box decoding inside the graph
    dummy = torch.randn(1, 3, size, size)
    last_err = None
    for opset in (17, 13, 11):
        try:
            kw = dict(input_names=["images"], output_names=["output"], opset_version=opset)
            try:
                torch.onnx.export(m, dummy, str(out_path), dynamo=False, **kw)
            except TypeError:                               # older PyTorch has no dynamo option
                torch.onnx.export(m, dummy, str(out_path), **kw)
            return opset
        except Exception as e:
            last_err = e
    raise last_err


def model_card(m):
    t, v = m["test"], m["val"]
    return f"""---
license: apache-2.0
pipeline_tag: object-detection
tags: [yolox, onnx, signature-detection, object-detection]
---
# Signature detection (YOLOX-s)

YOLOX-s fine-tuned to find signatures on document pages (1 class: signature).
Based on YOLOX by Megvii (Apache 2.0): https://github.com/Megvii-BaseDetection/YOLOX
Training data: AimaArshad/signature-detection-yolox (check that dataset's license before making this repo public).

## Results (box correct if IoU >= 0.5)
| Split | mAP@.50 | mAP@.50:.95 | mAP@.75 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Validation | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} | {v['mAP75']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |
| Test | {t['mAP50']:.3f} | {t['mAP50_95']:.3f} | {t['mAP75']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} |

Thresholds: confidence {m['conf_threshold']}, NMS {m['nms_threshold']} (tuned on validation).

## Files
- signature.onnx: deployment model (box decoding inside the graph)
- onnx_infer.py: stand-alone inference script (onnxruntime, opencv, numpy)
- best.pt: YOLOX checkpoint (not an Ultralytics file)
- history.json: per-epoch mAP and loss, metrics.json: final scores
- {EXP_NAME}.py: YOLOX config to rebuild the model from best.pt
"""


def patch_infer_script(src, dst, conf, nms, size):
    """Copy onnx_infer.py and set the tuned defaults at the top of the file."""
    txt = Path(src).read_text()
    txt = re.sub(r"^DEFAULT_CONF = [0-9.]+", f"DEFAULT_CONF = {conf}", txt, flags=re.M)
    txt = re.sub(r"^DEFAULT_NMS = [0-9.]+", f"DEFAULT_NMS = {nms}", txt, flags=re.M)
    txt = re.sub(r"^DEFAULT_SIZE = \(\d+, \d+\)", f"DEFAULT_SIZE = ({size}, {size})", txt, flags=re.M)
    Path(dst).write_text(txt)


def export_all(det, final_dir, hist_path, exp_file, metrics, size, data_dir):
    torch = det.torch
    shutil.rmtree(final_dir, ignore_errors=True)
    final_dir.mkdir(parents=True)
    plain = lambda d: json.loads(json.dumps(d, default=float))
    torch.save({"model": det.ckpt["model"], "arch": "YOLOX-s", "classes": CLASSES, "input_size": [size, size],
                "conf_threshold": metrics["conf_threshold"], "nms_threshold": metrics["nms_threshold"],
                "epoch": det.ckpt.get("epoch"), "metrics": plain(metrics)}, str(final_dir / "best.pt"))
    opset = export_onnx(det, final_dir / "signature.onnx", size)
    print("signature.onnx exported (opset", opset, ")")
    infer_src = HERE / "onnx_infer.py"
    if not infer_src.exists():
        raise FileNotFoundError(f"{infer_src} is missing. Keep onnx_infer.py next to train.py.")
    patch_infer_script(infer_src, final_dir / "onnx_infer.py", metrics["conf_threshold"], metrics["nms_threshold"], size)
    shutil.copy(hist_path, final_dir / "history.json")
    json.dump(plain(metrics), open(final_dir / "metrics.json", "w"), indent=2)
    shutil.copy(exp_file, final_dir / f"{EXP_NAME}.py")
    (final_dir / "README.md").write_text(model_card(metrics))

    # check that ONNX gives the same boxes as PyTorch
    spec = importlib.util.spec_from_file_location("onnx_infer", final_dir / "onnx_infer.py")
    oi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oi)
    sess = oi.load_session(str(final_dir / "signature.onnx"))
    images, _, img_dir, _ = load_split(data_dir, "test", 40)
    n_torch = matched = 0
    for im in images:
        img = cv2.imread(str(img_dir / im["file_name"]))
        a = det.predict(img, metrics["conf_threshold"], metrics["nms_threshold"])
        b = oi.detect(sess, img, metrics["conf_threshold"], metrics["nms_threshold"])
        n_torch += len(a)
        matched += sum(1 for pa in a if any(iou(pa, pb[:4]) > 0.90 for pb in b))
    metrics["onnx_parity"] = {"images": len(images), "pytorch_boxes": n_torch, "matched": matched}
    print(f"ONNX vs PyTorch: {matched}/{n_torch} boxes matched (IoU > 0.90)")
    json.dump(plain(metrics), open(final_dir / "metrics.json", "w"), indent=2)


# ----------------------------------------------------------------------------------------------
# Optional: Comet logging and Hugging Face upload
# ----------------------------------------------------------------------------------------------
def log_to_comet(args, history, metrics, final_dir):
    import comet_ml
    exp = comet_ml.Experiment(project_name=args.comet_project)      # reads COMET_API_KEY from the environment
    exp.set_name(f"yolox-s-signature-{time.strftime('%Y%m%d-%H%M')}")
    exp.log_parameters({"epochs_max": args.epochs, "batch_size": args.batch_size, "input_size": args.input_size,
                        "monitor": args.monitor, "patience": args.patience, "neg_ratio": args.neg_ratio,
                        "conf_threshold": metrics["conf_threshold"], "nms_threshold": metrics["nms_threshold"]})
    for e in history["epochs"]:
        exp.log_metric("val/mAP_0.5", e["ap50"], step=e["epoch"])
        exp.log_metric("val/mAP_0.5:0.95", e["ap50_95"], step=e["epoch"])
        if e.get("train_loss") is not None:
            exp.log_metric("train/loss", e["train_loss"], step=e["epoch"])
    for split in ("val", "test"):
        exp.log_metrics({f"{split}/{k}": v for k, v in metrics[split].items() if isinstance(v, (int, float))})
    exp.log_asset(str(final_dir / "metrics.json"))
    exp.end()
    print("Logged to Comet project:", args.comet_project)


def push_to_hub(args, final_dir):
    from huggingface_hub import HfApi
    api = HfApi()                                                   # reads HF_TOKEN from the environment
    api.create_repo(args.push_to_hub, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(final_dir), repo_id=args.push_to_hub, repo_type="model",
                      ignore_patterns=["__pycache__/*", "*.pyc"], commit_message="Add YOLOX-s signature detector")
    print("Uploaded: https://huggingface.co/" + args.push_to_hub)


# ----------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out, yolox_dir, data_dir = Path(args.out_dir).resolve(), Path(args.yolox_dir).resolve(), Path(args.data_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.quick_test:
        args.epochs, args.eval_interval = 2, 1
        max_train, max_val, eval_limit, sweep_limit = 1500, 300, 300, 300
    else:
        max_train = max_val = eval_limit = None
        sweep_limit = 1500

    import torch
    if not args.skip_train and not torch.cuda.is_available():
        sys.exit("No GPU found. Training needs a CUDA GPU (use --skip-train to only evaluate/export).")

    weights = setup_yolox(yolox_dir)
    get_data(data_dir, args.hf_dataset)
    ann = data_dir / "annotations"

    train_ann, val_ann = "instances_train2017.json", "instances_val2017.json"
    if args.quick_test:
        make_subset(ann / train_ann, ann / "instances_train_small.json", max_train)
        make_subset(ann / val_ann, ann / "instances_val_small.json", max_val)
        train_ann, val_ann = "instances_train_small.json", "instances_val_small.json"
    if args.neg_ratio > 0:
        neg_name = train_ann.replace(".json", "_neg.json")
        n = make_negatives(ann, train_ann, neg_name, data_dir / "train2017", args.neg_ratio, args.neg_max)
        print(f"Added {n} negative pages")
        train_ann = neg_name

    exp_file = yolox_dir / "exps" / "custom" / f"{EXP_NAME}.py"
    exp_file.parent.mkdir(parents=True, exist_ok=True)
    exp_file.write_text(
        EXP_TEMPLATE.replace("__NC__", str(len(CLASSES))).replace("__DATA__", str(data_dir))
        .replace("__TRAIN_ANN__", train_ann).replace("__VAL_ANN__", val_ann)
        .replace("__SIZE__", str(args.input_size)).replace("__EPOCHS__", str(args.epochs))
        .replace("__NOAUG__", "1" if args.quick_test else "5").replace("__EVAL__", str(args.eval_interval))
        .replace("__NAME__", EXP_NAME))

    yolox_out = yolox_dir / "YOLOX_outputs" / EXP_NAME
    info = {"epochs": args.epochs, "quick": args.quick_test, "bs": args.batch_size, "size": args.input_size,
            "neg": args.neg_ratio, "eval": args.eval_interval}
    info_path = out / "run_info.json"
    if not args.skip_train:
        if info_path.exists() and json.load(open(info_path)) != info:
            shutil.rmtree(yolox_out, ignore_errors=True)    # main settings changed -> start fresh
        json.dump(info, open(info_path, "w"))
        train(args, yolox_dir, exp_file, weights, yolox_out)

    best = yolox_out / "best_monitor_ckpt.pth"
    if not best.exists():
        best = yolox_out / "best_ckpt.pth"
    if not best.exists():
        sys.exit("No best checkpoint found. Train first.")
    det = Detector(exp_file, best)

    conf, nms, sweeps = tune_thresholds(det, data_dir, sweep_limit, args.precision_target)
    metrics = {"conf_threshold": conf, "nms_threshold": nms, "monitor": args.monitor, **sweeps}
    metrics["val"] = evaluate_split(det, data_dir, "val", eval_limit, conf, nms)
    metrics["test"] = evaluate_split(det, data_dir, "test", eval_limit, conf, nms)
    for s in ("val", "test"):
        r = metrics[s]
        print(f"{s:>5}: mAP@.50 {r['mAP50']:.3f} | mAP@.50:.95 {r['mAP50_95']:.3f} | mAP@.75 {r['mAP75']:.3f} | "
              f"P {r['precision']:.3f} | R {r['recall']:.3f} | F1 {r['f1']:.3f}")

    final_dir = out / "final_model"
    export_all(det, final_dir, yolox_out / "history.json", exp_file, metrics, args.input_size, data_dir)
    print("Saved to", final_dir)

    if args.comet:
        log_to_comet(args, json.load(open(yolox_out / "history.json")), metrics, final_dir)
    if args.push_to_hub:
        push_to_hub(args, final_dir)


if __name__ == "__main__":
    main()
