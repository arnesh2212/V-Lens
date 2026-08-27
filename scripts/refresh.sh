#!/bin/bash
# Pull everything down and rebuild the artifact locally.
bash /home/arnesg/vlm/scripts/pull.sh full
python3 /home/arnesg/vlm/paper/build.py
