#!/bin/bash
# Sequential GPU job queue; each line runs to completion before the next.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
run(){ echo "=== $(date +%H:%M:%S) $*" >> logs/queue.log; "$@" >> logs/queue.log 2>&1; echo "=== rc=$? $(date +%H:%M:%S)" >> logs/queue.log; }
