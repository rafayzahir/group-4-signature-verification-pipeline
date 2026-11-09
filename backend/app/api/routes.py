import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import redis
from celery.result import AsyncResult
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.celery_app import REDIS_URL, celery_app

router = APIRouter()

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 5 * 1024 * 1024
JOB_TTL = 86400

def job_key(job_id: str) -> str:
    return f"signature_job:{job_id}"


async def save_signature(file: UploadFile, destination: Path):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: Only JPG and PNG images are accepted."
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image size must not exceed 5 MB.")

    try:
        with Image.open(BytesIO(content)) as img:
            expected_format = (
                "JPEG" if file.content_type == "image/jpeg" else "PNG"
            )

            if img.format != expected_format:
                raise HTTPException(
                    status_code=400,
                    detail="Image content does not match its file type."
                )

            img.verify()

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid or readable image."
        )

    destination.write_bytes(content)


@router.post("/verify")
async def verify_signature(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    job_id = str(uuid4())

    reference_ext = ".jpg" if image1.content_type == "image/jpeg" else ".png"
    submitted_ext = ".jpg" if image2.content_type == "image/jpeg" else ".png"

    reference_path = UPLOAD_DIR / f"{job_id}_reference{reference_ext}"
    submitted_path = UPLOAD_DIR / f"{job_id}_submitted{submitted_ext}"

    try:
        await save_signature(image1, reference_path)
        await save_signature(image2, submitted_path)

        redis_client.setex(
            job_key(job_id),
            JOB_TTL,
            json.dumps({
                "job_id": job_id,
                "status": "pending",
                "reference_image": str(reference_path),
                "submitted_image": str(submitted_path)
            })
        )

        celery_app.send_task(
            "app.tasks.verify_signatures",
            args=[str(reference_path), str(submitted_path)],
            task_id=job_id
        )

    except Exception:
        redis_client.delete(job_key(job_id))
        reference_path.unlink(missing_ok=True)
        submitted_path.unlink(missing_ok=True)
        raise

    finally:
        await image1.close()
        await image2.close()

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Signature verification job created successfully."
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    metadata = redis_client.get(job_key(job_id))

    if metadata is None:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    task = AsyncResult(job_id, app=celery_app)

    status_map = {
        "PENDING": "pending",
        "STARTED": "processing",
        "SUCCESS": "completed",
        "FAILURE": "failed",
        "RETRY": "processing",
        "REVOKED": "failed"
    }

    status = status_map.get(task.state, "processing")

    return {
        "job_id": job_id,
        "status": status,
        "message": "Job status retrieved successfully."
    }


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    metadata = redis_client.get(job_key(job_id))

    if metadata is None:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    task = AsyncResult(job_id, app=celery_app)

    if task.failed():
        raise HTTPException(
            status_code=500,
            detail="Signature verification task failed."
        )

    if not task.successful():
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed yet. Current status: {task.state}"
        )

    result = task.result

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Worker returned an invalid result."
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "score": result.get("similarity_score"),
        "result": result.get("result")
    }
