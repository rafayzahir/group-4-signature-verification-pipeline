# onnx_infer.py - signature detection with ONNX Runtime (YOLOX-s exported by the training notebook)
# Needs only:  pip install onnxruntime opencv-python numpy
#
#   python onnx_infer.py --image page.jpg
#   python onnx_infer.py --image pages_folder --out results
#   python onnx_infer.py --image page.jpg --conf 0.6 --nms 0.4
#
# Prints one JSON line per image:
#   {"image": "...", "signatures": [{"box": [x1, y1, x2, y2], "score": 0.97}]}
import argparse, glob, json, os
import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_CONF = 0.9      # confidence threshold tuned on the validation set
DEFAULT_NMS = 0.3        # NMS IoU threshold tuned on the validation set
DEFAULT_SIZE = (640, 640)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_session(model_path="signature.onnx"):
    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    return ort.InferenceSession(model_path, providers=providers)


def _input_size(session):
    shp = session.get_inputs()[0].shape               # [1, 3, H, W]
    if isinstance(shp[2], int) and isinstance(shp[3], int):
        return (shp[2], shp[3])
    return DEFAULT_SIZE


def preprocess(img_bgr, size):
    # YOLOX convention: keep aspect ratio, pad with 114, BGR, values 0-255, NCHW
    h, w = img_bgr.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    return blob, r


def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= thr]
    return keep


def postprocess(output, ratio, img_shape, conf, nms_thr):
    p = output[0]                                      # (N, 5 + classes): cx, cy, w, h, objectness, class scores
    scores = p[:, 4] * p[:, 5:].max(axis=1)
    m = scores >= conf
    p, scores = p[m], scores[m]
    if len(p) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    boxes = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                      p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], axis=1) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])
    keep = nms(boxes, scores, nms_thr)
    return np.concatenate([boxes[keep], scores[keep, None]], axis=1)


def detect(session, img_bgr, conf=DEFAULT_CONF, nms_thr=DEFAULT_NMS):
    # Returns an array of [x1, y1, x2, y2, score] in original image pixels
    blob, ratio = preprocess(img_bgr, _input_size(session))
    output = session.run(None, {session.get_inputs()[0].name: blob})[0]
    return postprocess(output, ratio, img_bgr.shape, conf, nms_thr)


def draw(img_bgr, dets):
    out = img_bgr.copy()
    for x1, y1, x2, y2, s in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(out, f"signature {s:.2f}", (int(x1), max(int(y1) - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="Signature detection (ONNX)")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "signature.onnx"))
    ap.add_argument("--image", required=True, help="image file or folder of images")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--nms", type=float, default=DEFAULT_NMS)
    ap.add_argument("--out", default=None, help="folder to save annotated images (optional)")
    args = ap.parse_args()
    session = load_session(args.model)
    if os.path.isdir(args.image):
        paths = sorted(p for p in glob.glob(os.path.join(args.image, "*")) if p.lower().endswith(IMG_EXT))
    else:
        paths = [args.image]
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(json.dumps({"image": p, "error": "could not read image"}))
            continue
        dets = detect(session, img, args.conf, args.nms)
        sigs = [{"box": [round(float(v), 1) for v in d[:4]], "score": round(float(d[4]), 4)} for d in dets]
        print(json.dumps({"image": p, "signatures": sigs}))
        if args.out:
            cv2.imwrite(os.path.join(args.out, os.path.basename(p)), draw(img, dets))


if __name__ == "__main__":
    main()
