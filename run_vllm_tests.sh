#!/bin/bash
# Run the vLLM test suite inside the conda env.
#
# By default creates a fresh timestamped run dir under test-logs/.
# Pass an existing dir as the first argument to reuse it; a stale vllm.log
# there is removed first.
#
# Usage:
#   ./run_vllm_tests.sh                 # fresh test-logs/<timestamp>/ dir
#   ./run_vllm_tests.sh test-logs/foo   # reuse test-logs/foo/, clearing vllm.log
#
# Submit as a non-interactive LSF job:
#   bsub -J vllm-tests -G grp_big_job_lease \
#        -gpu "num=2/task:mode=exclusive_process" -n 1 -M 100G -W 360 \
#        -o test-logs/lsf.%J.out -e test-logs/lsf.%J.err \
#        ./run_vllm_tests.sh

set -euo pipefail
# Note: pipefail (above) makes `make test-vllm | tee` exit with pytest's
# status rather than tee's, so a test failure propagates to the LSF job.

# cd to the repo root (dir holding this script) so relative paths resolve
# regardless of where the job starts.
cd "$(dirname "$(readlink -f "$0")")"

# --- Load the conda environment ---
source /opt/share/miniforge/etc/profile.d/conda.sh
conda activate /proj/comodel/antonp/venv

# --- Resolve the run dir ---
if [ "$#" -ge 1 ]; then
    RUN="$1"
    mkdir -p "$RUN"
    # Remove a stale log so this run doesn't append to a previous one.
    if [ -f "$RUN/vllm.log" ]; then
        echo "Removing existing $RUN/vllm.log"
        rm -f "$RUN/vllm.log"
    fi
else
    RUN="test-logs/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$RUN"
fi

echo "Run dir: $RUN"

# --- Run the vLLM tests ---
make test-vllm \
    PYTEST_FLAGS="-v -s --tb=short --junitxml=$RUN/vllm.xml" \
    2>&1 | tee "$RUN/vllm.log"
