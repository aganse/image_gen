#!/bin/bash
set -e

# Convention (project-agnostic): TRAINING_SCRIPT=<name> runs /app/train_<name>.py
# with parameter overrides from params_<name>.json.
if [ -z "$TRAINING_SCRIPT" ]; then
    echo "entrypoint: TRAINING_SCRIPT is not set (expected e.g. TRAINING_SCRIPT=vae" \
         "to run train_vae.py)" >&2
    exit 1
fi

SCRIPT="train_${TRAINING_SCRIPT}.py"
HP_FILE="/opt/ml/input/config/hyperparameters.json"
PARAMS_FILE="/opt/ml/input/data/params/params_${TRAINING_SCRIPT}.json"

if [ ! -f "/app/$SCRIPT" ]; then
    echo "entrypoint: /app/$SCRIPT not found" >&2
    exit 1
fi

# For local Docker: if params file is mounted but HP file is absent, create it
# so the script's param-loading logic works the same in all container contexts.
if [ ! -f "$HP_FILE" ] && [ -f "$PARAMS_FILE" ]; then
    mkdir -p "$(dirname "$HP_FILE")"
    BLOB=$(base64 -w0 "$PARAMS_FILE" 2>/dev/null || base64 "$PARAMS_FILE")
    printf '{"params": "%s"}' "$BLOB" > "$HP_FILE"
fi

echo "Running: python3 $SCRIPT"
exec python3 /app/$SCRIPT
