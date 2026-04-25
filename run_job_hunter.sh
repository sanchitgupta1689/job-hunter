#!/bin/bash
# Daily Job Hunter — called by crontab at 9 AM IST (3:30 AM UTC)

PROJECT_DIR="/Users/i0j00vh/Documents/job_hunter"
LOG_FILE="$PROJECT_DIR/cron.log"

echo "========================================" >> "$LOG_FILE"
echo "Job Hunter started: $(date)" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1
source "$PROJECT_DIR/venv/bin/activate"

python main.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "Job Hunter finished: $(date) (exit $EXIT_CODE)" >> "$LOG_FILE"
