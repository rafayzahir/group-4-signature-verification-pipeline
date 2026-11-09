
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

# Temporary in-memory job storage.
# This will be replaced with the team's shared result storage.
jobs = {}

# Directory where uploaded signature images will be saved.
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image formats and maximum file size.
ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


async def save_signature(file: UploadFile, destination: Path):
    # Check the uploaded file's content type.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: Only JPG and PNG images are accepted."
        )

    # Read the file while enforcing the size limit.
    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 5 MB."
        )

    # Verify that the uploaded file is a valid, readable image.
    try:
        with Image.open(BytesIO(content)) as img:

            expected_format = (
                "JPEG" if file.content_type == "image/jpeg" else "PNG"
            )

            # Ensure actual image format matches the uploaded file type.
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

    # Save the validated image.
    destination.write_bytes(content)


# --------------------------------------------------
# POST /verify
# Upload two signature images and create a job.
# --------------------------------------------------

@router.post("/verify")
async def verify_signature(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    job_id = str(uuid4())

    # Preserve the correct image extensions.
    reference_ext = (
        ".jpg" if image1.content_type == "image/jpeg" else ".png"
    )
    submitted_ext = (
        ".jpg" if image2.content_type == "image/jpeg" else ".png"
    )

    reference_path = UPLOAD_DIR / f"{job_id}_reference{reference_ext}"
    submitted_path = UPLOAD_DIR / f"{job_id}_submitted{submitted_ext}"

    try:
        # Save both uploaded signature images.
        await save_signature(image1, reference_path)
        await save_signature(image2, submitted_path)

        # Create a temporary job record.
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "reference_image": str(reference_path),
            "submitted_image": str(submitted_path),
            "result": None
        }

    except Exception:
        # Remove partially uploaded files if anything goes wrong.
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


# --------------------------------------------------
# GET /status/{job_id}
# Retrieve the current job status.
# --------------------------------------------------

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):

    job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found."
        )

    return {
        "job_id": job_id,
        "status": job["status"],
        "message": "Job status retrieved successfully."
    }


# --------------------------------------------------
# GET /result/{job_id}
# Retrieve the completed verification result.
# --------------------------------------------------

@router.get("/result/{job_id}")
async def get_job_result(job_id: str):

    job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found."
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed yet. Current status: {job['status']}"
        )

    return {
    "job_id": job_id,
    "status": job["status"],
    "score": job["result"]
}
