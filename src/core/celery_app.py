"""Celery application configuration for production workloads.

Lightweight module imported by FastAPI to enqueue tasks and inspect status.
Heavy dependencies remain isolated within worker-imported modules.
"""
from celery import Celery

from src.core.config import settings

# ! store the final status in postgres sql not in redis
# ! create worker for heave task and light tasks

celery_app = Celery(
    "ingestion",
    broker=settings.CELERY_BROKER_URL,        # Job queue (Redis)
    backend=settings.CELERY_RESULT_BACKEND,   # Status & result store (Redis)
    include=["src.services.tasks"],          # Loaded exclusively by workers
)

celery_app.conf.update(
    # Strict Serialization Security: Prevents Remote Code Execution (RCE)
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    timezone="UTC",
    enable_utc=True,
    
    # Task State Tracking
    task_track_started=True,
    
    # Reliability & Replay Guarantees
    task_acks_late=True,
    task_reject_on_worker_lost=True,  # Rejects task if worker dies unexpectedly (OOM/Crash)
    
    # Load Balancing for Long-Running Tasks
    worker_prefetch_multiplier=1,
    
    # Memory Leak Prevention: Restarts worker process after 100 tasks to release RAM
    worker_max_tasks_per_child=100,
    
    # Timeouts and Execution Guards
    task_soft_time_limit=600,   # Raises SoftTimeLimitExceeded inside Python (10 mins)
    task_time_limit=720,        # Hard SIGKILL execution guard (12 mins)
    
    # [Critical Fix] Must be greater than task_time_limit to prevent duplicate execution
    broker_transport_options={'visibility_timeout': 1000}, 



    
    # Redis RAM Management: Clean up result metadata after 1 hour
    result_expires=3600,
    
    # Network Resiliency
    broker_connection_retry_on_startup=True,
    broker_pool_limit=10,
)