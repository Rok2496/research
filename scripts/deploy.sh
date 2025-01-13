#!/bin/bash

# Load deployment configuration
DEPLOY_CONFIG="configs/deployment_config.yaml"

# Parse arguments
ENV="production"
PORT=8000
WORKERS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --workers) WORKERS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Setup SSL certificates if enabled
if [ "$ENV" = "production" ]; then
    if [ ! -f "/etc/ssl/certs/cert.pem" ]; then
        echo "Generating SSL certificates..."
        openssl req -x509 -newkey rsa:4096 -nodes \
            -out /etc/ssl/certs/cert.pem \
            -keyout /etc/ssl/private/key.pem \
            -days 365 \
            -subj "/CN=chexpert.local"
    fi
fi

# Build and run Docker containers
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d

# Initialize monitoring
python -m core.monitoring.setup \
    --config $DEPLOY_CONFIG \
    --env $ENV

# Start application
if [ "$ENV" = "production" ]; then
    gunicorn --config deployment/serving/wsgi.py \
        --certfile=/etc/ssl/certs/cert.pem \
        --keyfile=/etc/ssl/private/key.pem \
        --bind 0.0.0.0:$PORT \
        --workers $WORKERS \
        deployment.serving.flask_app:app
else
    python deployment/serving/flask_app.py
fi
