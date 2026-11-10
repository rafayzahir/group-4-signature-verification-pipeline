# Runs YOLOX's own tools/train.py and adds: early stopping on validation mAP,
# best-weights saving by that metric, and a history.json log.
import os, sys, json, runpy
import torch
from yolox.core.trainer import Trainer

MONITOR    = os.environ.get("ES_MONITOR", "ap50")
PATIENCE   = int(os.environ.get("ES_PATIENCE", "8"))
MIN_EPOCHS = int(os.environ.get("ES_MIN_EPOCHS", "15"))
MIN_DELTA  = float(os.environ.get("ES_MIN_DELTA", "0.0"))
HIST_PATH  = os.environ["ES_HISTORY"]

os.makedirs(os.path.dirname(HIST_PATH), exist_ok=True)   # YOLOX creates this folder only later
STATE = {"epochs": []}
if os.path.exists(HIST_PATH):                       # resuming after a disconnect
    STATE = json.load(open(HIST_PATH))
STATE.update(monitor=MONITOR, patience=PATIENCE, min_epochs=MIN_EPOCHS, min_delta=MIN_DELTA,
             finished=False, stopped_early=False)

def _save():
    tmp = HIST_PATH + ".tmp"
    json.dump(STATE, open(tmp, "w"), indent=2)
    os.replace(tmp, HIST_PATH)

def _recompute():
    best, best_ep, bad = -1.0, 0, 0
    for e in sorted(STATE["epochs"], key=lambda e: e["epoch"]):
        if e[MONITOR] > best + MIN_DELTA:
            best, best_ep, bad = e[MONITOR], e["epoch"], 0
        else:
            bad += 1
    STATE.update(best=best, best_epoch=best_ep, bad_evals=bad)

_recompute()
_save()

_orig_eval_and_save = Trainer.evaluate_and_save_model

def evaluate_and_save_model(self):
    exp, real_eval, got = self.exp, self.exp.eval, {}
    def capture(*a, **k):                           # grab the mAP numbers YOLOX computes
        res = real_eval(*a, **k)
        m = res[0] if (isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], (tuple, list))) else res
        got["ap50_95"], got["ap50"] = float(m[0]), float(m[1])
        return res
    exp.eval = capture
    try:
        _orig_eval_and_save(self)                   # YOLOX evaluates + saves last_epoch_ckpt.pth
    finally:
        exp.__dict__.pop("eval", None)
    if "ap50" not in got:
        return
    epoch = self.epoch + 1
    try:
        loss = float(self.meter["total_loss"].avg)
    except Exception:
        loss = None
    entry = {"epoch": epoch, "ap50": got["ap50"], "ap50_95": got["ap50_95"], "train_loss": loss}
    STATE["epochs"] = [e for e in STATE["epochs"] if e["epoch"] != epoch] + [entry]
    _recompute()
    if STATE["best_epoch"] == epoch:                # new best -> keep these weights (model only, small file)
        ck = torch.load(os.path.join(self.file_name, "last_epoch_ckpt.pth"), map_location="cpu", weights_only=False)
        torch.save({"model": ck["model"], "epoch": epoch, "ap50": got["ap50"], "ap50_95": got["ap50_95"]},
                   os.path.join(self.file_name, "best_monitor_ckpt.pth"))
    if epoch >= MIN_EPOCHS and STATE["bad_evals"] >= PATIENCE:
        STATE["stopped_early"] = True
    _save()
    print(f"[EarlyStopping] epoch {epoch:>3} | mAP50 {got['ap50']:.4f} | mAP50:95 {got['ap50_95']:.4f} | "
          f"best {MONITOR} {STATE['best']:.4f} (epoch {STATE['best_epoch']}) | "
          f"no-improvement {STATE['bad_evals']}/{PATIENCE}", flush=True)

def train_in_epoch(self):                           # same loop as YOLOX, plus a stop check
    for self.epoch in range(self.start_epoch, self.max_epoch):
        self.before_epoch()
        self.train_in_iter()
        self.after_epoch()
        if STATE["stopped_early"]:
            print(f"[EarlyStopping] STOP: no new best {MONITOR} for {PATIENCE} evaluations. "
                  f"Best = {STATE['best']:.4f} at epoch {STATE['best_epoch']}", flush=True)
            break
    STATE["finished"] = True
    _save()

Trainer.evaluate_and_save_model = evaluate_and_save_model
Trainer.train_in_epoch = train_in_epoch
print(f"[EarlyStopping] watching {MONITOR} | patience {PATIENCE} | min epochs {MIN_EPOCHS} | min delta {MIN_DELTA}", flush=True)

train_py = os.path.join(os.environ["ES_YOLOX_DIR"], "tools", "train.py")
sys.argv = [train_py] + sys.argv[1:]
runpy.run_path(train_py, run_name="__main__")
