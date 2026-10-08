#!/bin/bash
# Restart the queue runner if it ever stops.
#
# The watchdog has died twice: once to a pkill whose pattern matched the ssh
# command issuing it, once to a killed session.  Each time the card sat idle
# until someone looked.  This checks every half minute and costs nothing.
cd /workspace/vlm
LOG=logs/supervise.log
while true; do
  if ! pgrep -f "watchdo[g].sh" >/dev/null 2>&1; then
    echo "$(date +%H:%M:%S) watchdog was down; restarting" >> "$LOG"
    setsid nohup bash watchdog.sh </dev/null >/dev/null 2>&1 &
  fi
  sleep 30
done
