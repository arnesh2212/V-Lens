#!/bin/bash
# push code to GPU box
rsync -az --delete -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
  /home/arnesg/vlm/code/ root@217.18.55.95:code/ 2>&1 | tail -3
