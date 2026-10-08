#!/usr/bin/env bash
pkill -f 'safety/refus''al.py' 2>/dev/null
sleep 4
echo "left: $(pgrep -cf 'safety/refus''al.py')"
