#!/bin/bash
# once the 27B patching run ends, use its share of the GPU for the two short 8B sandbagging jobs
cd /workspace/temp/vlm; source /venv/main/bin/activate
export VLENS_DATA="/workspace/temp/vlm/data_root/VLM V-Lens Final Dataset Complete"
while pgrep -f "patch_causal.py --repo Qwen/Qwen3.5-27B" >/dev/null; do sleep 60; done
echo "$(date +%m-%d\ %H:%M) patch_causal done, starting 8B sandbagging side jobs"
head -2 queue/main_one.q | while IFS= read -r line; do
  case "$line" in *Qwen3-VL-8B-Instruct*sand*) ;; *) continue;; esac
  log=logs/side_sand.$(date +%m%d-%H%M%S).log
  echo "$(date +%H:%M) run: $line"
  if bash -c "$line" > "$log" 2>&1; then echo "$line" >> queue/main_one.q.done.lines; echo "  ok -> $log"; else echo "  FAILED -> $log"; fi
done
echo "$(date +%H:%M) side jobs finished"
