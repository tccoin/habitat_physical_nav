#!/bin/bash

# ==========================================
# Configuration Constants
# ==========================================
# The box has 256 cores; nproc reports 4 only because OMP_NUM_THREADS=4 is set.
# The bottleneck is per-file JPEG writes to NFS (80% of wall-clock), not the
# simulator, so this scales well past any CPU limit -- measured 8.69x at 16 and
# 13.20x at 32. See dump/regen_readiness/FINDINGS.md.
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-32}"
MAX_LENGTH=300
# Reject an episode that stays off its reference path this long. Arming this
# costs 2.64% of otherwise-good episodes; leaving it off ships the ones that
# grind against geometry. Escape rejection (the joint_z fall detector) is always
# on and is not this knob -- see dump/drive_failure_detection/FINDINGS.md.
REJECT_REF_MISS_S="${REJECT_REF_MISS_S:-1.0}"
# 25 Hz -> dt = 0.04s = exactly 5 of habitat's fixed 0.008s Bullet steps. Off-grid
# rates (the old FPS=30 -> dt=0.03333s = 4.1667 steps) make the physics time
# delivered per tick alternate between 0.032s and 0.040s, i.e. a 24% per-tick
# velocity ripple baked into every recorded pose. See HabitatRobotSim.step.
#
# Whatever this is set to is now written into every metadata record (the
# "recording" block -- see continuous_demos.recording_metadata), so consumers
# read the rate instead of being told it. Overridable from the environment for
# a deliberate off-grid comparison run; the corpus will say so.
FPS="${FPS:-25}"
IMAGE_EXT="jpeg"

# ==========================================
# Paths
# ==========================================
# Resolve everything from this script's own location. Previously TARGET_SCRIPT
# was relative to scripts/ while OUTPUT_DIR was relative to the repo root, so the
# script only worked from scripts/ and silently wrote its corpus into
# scripts/output_25hz -- and running it from the repo root failed outright.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHON_BIN="/workspace/conda/envs/vln/bin/python3"
export TARGET_SCRIPT="$SCRIPT_DIR/generate_pose2d_dataset.py"
export INPUT_DIR="/Projects/continuous_demos/habitat-web/objectnav/objectnav_mp3d_thda_70k/train/content"
# The 30 Hz pre-schema corpus lives in output/. Generation appends, and the
# generator now refuses to append records to a shard recorded at a different
# rate (or to a pre-schema shard, which carries no rate at all), so a
# regeneration must write somewhere fresh -- hence the new default. Set
# OUTPUT_DIR in the environment to resume a run in progress.
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output_25hz}"
export LOGS_DIR="$OUTPUT_DIR/logs"

# Export variables needed by the xargs subshell
export MAX_LENGTH FPS IMAGE_EXT REJECT_REF_MISS_S

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
        --reject-ref-miss-s "$REJECT_REF_MISS_S" \
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