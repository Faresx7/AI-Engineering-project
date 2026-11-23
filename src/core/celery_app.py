"""Celery application configuration for production workloads.

Lightweight module imported by FastAPI to enqueue tasks and inspect status.
Heavy dependencies remain isolated within worker-imported modules.
"""
from datetime import timedelta
from celery import Celery


from src.core.config import settings

# ! store the final status in postgres sql not in redis
# ! create worker for heavy task and light tasks

# Initializes the Celery application instance and links it to Redis and worker task modules
celery_app = Celery("ingestion",
                    broker=settings.CELERY_BROKER_URL,        # Job queue (Redis)
                    backend=settings.CELERY_RESULT_BACKEND,   # Status & result store (Redis)
                    include=["src.services.tasks"],          # Loaded exclusively by workers
                   )

celery_app.conf.update(# Strict Serialization Security: Prevents Remote Code Execution (RCE)
                       # Formats outgoing task arguments as plain JSON instead of unsafe pickle
                       task_serializer="json",
                       # Formats task return values and exception metadata as plain JSON
                       result_serializer="json",
                       # Rejects any incoming message that is not encoded in JSON
                       accept_content=["json"],

                       # Standardizes all scheduled tasks and internal timestamps to UTC
                       timezone="UTC",
                       # Forces Celery to store and interpret all datetimes in UTC
                       enable_utc=True,

                       # Task State Tracking
                       # Reports "STARTED" state as soon as a worker picks up a task from the queue
                       task_track_started=True,

                       # Reliability & Replay Guarantees
                       # Acknowledges and deletes the task from Redis only AFTER it finishes executing
                       task_acks_late=True,
                       # Re-queues an unacknowledged task if the worker process crashes or gets OOM-killed
                       task_reject_on_worker_lost=True,  # Rejects task if worker dies unexpectedly (OOM/Crash)

                       # Load Balancing for Long-Running Tasks
                       # Forces each worker to reserve only 1 task at a time so heavy jobs do not pile up on one worker
                       worker_prefetch_multiplier=1,

                       # Memory Leak Prevention: Restarts worker process after 100 tasks to release RAM
                       # Replaces the worker child process every 100 completed tasks to reclaim fragmented memory
                       worker_max_tasks_per_child=100,

                       # Timeouts and Execution Guards
                       # Sends SoftTimeLimitExceeded after 700s so the task can clean up and roll back gracefully
                       task_soft_time_limit=700,   # Raises SoftTimeLimitExceeded inside Python
                       # Force-kills the worker process with SIGKILL after 800s if soft cleanup hangs
                       task_time_limit=800,        # Hard SIGKILL execution guard

                       # Must be greater than task_time_limit to prevent duplicate execution
                       # Tells Redis how long to wait for an ack before redelivering an unconfirmed task to another worker
                       broker_transport_options={"visibility_timeout": 2500},

                       # Redis RAM Management: Clean up result metadata after 1 hour
                       # Automatically deletes stored task results from the Redis backend after 3600 seconds
                       result_expires=3600,

                       # Network Resiliency
                       # Keeps retrying the initial broker connection on worker startup if Redis is not ready yet
                       broker_connection_retry_on_startup=True,
                       # Caps the maximum number of open TCP connections pooled to the Redis broker at 10
                       broker_pool_limit=10,

                       # Periodic Maintenance Schedule (Celery Beat)
                       # Defines recurring background tasks triggered automatically by Celery Beat
                       beat_schedule={# Periodic sweep job entry for removing abandoned Qdrant generations
                                      "purge-orphaned-qdrant-generations": {# Registered Celery task name to dispatch
                                                                            "task": "rag.purge_orphaned_generations",
                                                                            # Triggers the sweep automatically every 75 minutes
                                                                            "schedule": timedelta(minutes=75),
                                                                            # Sets message expiration so missed ticks do not pile up if workers were offline
                                                                            "options": {"expires": timedelta(minutes=70)},
                                                                           },
                                     },
                      )