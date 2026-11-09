import json
import re
from pathlib import Path

import cv2

from worker.app.celery_app import celery_app
from detection.onnx_infer import load_session, detect, draw
from similarity.similarity import compute_similarity, THRESHOLD


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_detection_session = None


def get_detection_session():
    global _detection_session

    if _detection_session is None:
        model_path = Path(__file__).resolve().parents[2] / "detection" / "signature.onnx"
        _detection_session = load_session(str(model_path))

    return _detection_session


@celery_app.task(
    bind=True,
    name="worker.process_signature_job"
)
def process_signature_job(
    self,
    job_id,
    document_path,
    reference_signature_path
):
    document = Path(document_path)
    reference = Path(reference_signature_path)

    if not document.is_file():
        raise FileNotFoundError(f"Document not found: {document}")

    if not reference.is_file():
        raise FileNotFoundError(f"Reference signature not found: {reference}")

    safe_job_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(job_id))

    if not safe_job_id:
        raise ValueError("Invalid job_id")

    job_dir = RESULTS_DIR / safe_job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(document))

    if image is None:
        raise ValueError(f"Unable to read document image: {document}")

    session = get_detection_session()
    detections = detect(session, image)

    verification_results = []

    for index, (x1, y1, x2, y2, confidence) in enumerate(detections, start=1):
        height, width = image.shape[:2]

        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        crop_path = job_dir / f"signature_{index}.png"

        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"Could not save signature crop: {crop_path}")

        score = compute_similarity(str(crop_path), str(reference))

        verification_results.append({
            "signature_index": index,
            "bounding_box": [x1, y1, x2, y2],
            "detection_confidence": round(float(confidence), 4),
            "similarity_score": score,
            "is_match": score >= THRESHOLD,
            "crop_path": str(crop_path)
        })

    annotated_image = draw(image, detections)
    annotated_path = job_dir / "annotated.jpg"

    if not cv2.imwrite(str(annotated_path), annotated_image):
        raise RuntimeError("Could not save annotated document")

    best_match = max(
        verification_results,
        key=lambda item: item["similarity_score"],
        default=None
    )

    if not verification_results:
        status = "no_signature_detected"
        match = False
    else:
        status = "completed"
        match = best_match["is_match"]

    result = {
        "job_id": job_id,
        "status": status,
        "match": match,
        "threshold": THRESHOLD,
        "best_similarity_score": (
            best_match["similarity_score"] if best_match else None
        ),
        "message": (
            "Signature verified successfully" if match
            else "No matching signature found" if verification_results
            else "No signature detected in document"
        ),
        "document_path": str(document),
        "reference_signature_path": str(reference),
        "detected_signatures": verification_results,
        "annotated_image_path": str(annotated_path)
    }

    result_file = RESULTS_DIR / f"{safe_job_id}.json"

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    result["result_file"] = str(result_file)

    return result
