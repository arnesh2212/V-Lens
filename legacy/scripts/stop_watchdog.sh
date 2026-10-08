#!/bin/bash
# Stop the queue runner by pid.
#
# Never `pkill -f watchdog`: that pattern matches every command line that
# mentions the log file, including the ssh command running the pkill, so it
# reliably kills the session and leaves the queue running unattended.
P=logs/watchdog.pid
if [ -f "$P" ] && kill -0 "$(cat "$P")" 2>/dev/null; then
  kill "$(cat "$P")" && echo "stopped pid $(cat "$P")"
else
  echo "not running"
fi
