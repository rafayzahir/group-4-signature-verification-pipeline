import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def get_job_result(job_id: str):
    result_file = RESULTS_DIR / f"{job_id}.json"

    if not result_file.exists():
        return None

    with open(result_file, "r") as file:
        return json.load(file)
    