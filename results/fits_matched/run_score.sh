#!/bin/bash
# fetch the tuned lens fitted at V-Read's token budget, relabel it, and read the stored states through it
cd /home/arnesg/vlm
D=results/fits_matched
[ -f $D/maps/7d15ba38e9b1ea14.pt.spec ] || scp -q root@217.18.55.118:/workspace/maps/7d15ba38e9b1ea14.pt root@217.18.55.118:/workspace/maps/7d15ba38e9b1ea14.pt.spec $D/maps/
python3 -c "
import torch;p='$D/maps/7d15ba38e9b1ea14.pt';d=torch.load(p,map_location='cpu')
d['maps']['v_read']=d['maps']['tuned']
torch.save(d,p);print('relabelled',list(d['maps']))"
export VLENS_DATA="$PWD/VLM V-Lens Final Dataset Complete" SIMONSAYS_DATA="$PWD/VLM V-Lens Final Dataset Complete"
python3 tools/layer_readouts.py read --model Qwen3-VL-8B-Instruct --maps $D/maps --out $D/readouts --families openimages_orig_auth openimages_edit_auth openimages_orig_eval nuscenes_edit_auth vins_orig_auth vins_edit_auth gemini_orig_auth wmdp_bio_sand wmdp_chem_sand wmdp_cybe_sand
echo EXIT $?
