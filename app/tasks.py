import json
from pathlib import Path
from app.celery_app import celery_app

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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
        raise FileNotFoundError(
            f"Document not found: {document}"
        )

    if not reference.is_file():
        raise FileNotFoundError(
            f"Signature not found: {reference}"
        )

    result = {
        "job_id": job_id,
        "status": "validated",
        "message": "Job validated successfully",
        "document_path": str(document),
        "reference_signature_path": str(reference)
    }

    result_file = RESULTS_DIR / f"{job_id}.json"

    with open(result_file, "w") as file:
        json.dump(result, file, indent=4)

    return result