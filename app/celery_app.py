import os
from celery import Celery
from dotenv import load_dotenv

# Env file location configuration
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Celery App Initialization
celery_app = Celery(
    "signature_worker",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    include=["app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)
