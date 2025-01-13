import multiprocessing
import os

# Gunicorn config
bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
worker_class = "gthread"
timeout = 120

# SSL config (for production)
keyfile = os.environ.get("SSL_KEYFILE")
certfile = os.environ.get("SSL_CERTFILE")

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Worker configurations
worker_tmp_dir = "/dev/shm"
max_requests = 1000
max_requests_jitter = 50

# Process naming
proc_name = "chexpert_advanced"
