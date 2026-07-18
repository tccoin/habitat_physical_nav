#!/bin/bash

# ==========================================
# Configuration Constants
# ==========================================
MAX_CONCURRENT_JOBS=24  # Adjust based on your CPU cores
MAX_LENGTH=300
FPS=30
IMAGE_EXT="jpeg"

# ==========================================
# Paths
# ==========================================
export PYTHON_BIN="/workspace/conda/envs/vln/bin/python3"
export TARGET_SCRIPT="generate_pose2d_dataset.py"
export INPUT_DIR="/Projects/continuous_demos/habitat-web/objectnav/objectnav_mp3d_thda_70k/train/content"
export OUTPUT_DIR="output"
export LOGS_DIR="$OUTPUT_DIR/logs"

# Export variables needed by the xargs subshell
export MAX_LENGTH FPS IMAGE_EXT

# Ensure output and log directories exist before starting
mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# ==========================================
# Worker Function
# ==========================================
process_scene() {
    local input_file="$1"
    
    # Extract filename without the .json.gz extension
    local base_name=$(basename "$input_file" .json.gz)
    
    # Define unique log path for this specific job
    local log_file="$LOGS_DIR/${base_name}_failed.log"
    
    # Execute the python script
    "$PYTHON_BIN" "$TARGET_SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --max-length "$MAX_LENGTH" \
        --fps "$FPS" \
        --image-ext "$IMAGE_EXT" \
        --log-file "$log_file" \
        "$input_file"
}

# Export the function so xargs can invoke it in a spawned bash shell
export -f process_scene

# ==========================================
# Execution
# ==========================================
echo "Starting parallel processing with $MAX_CONCURRENT_JOBS workers..."

# Find all json.gz files and pipe them safely to xargs
find "$INPUT_DIR" -name "*.json.gz" -print0 | \
    xargs -0 -n 1 -P "$MAX_CONCURRENT_JOBS" bash -c 'process_scene "$@"' _

echo "Processing complete. Logs are stored in $LOGS_DIR"