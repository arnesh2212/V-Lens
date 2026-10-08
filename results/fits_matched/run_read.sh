#!/bin/bash
cd /home/arnesg/vlm
export VLENS_DATA="$PWD/VLM V-Lens Final Dataset Complete" SIMONSAYS_DATA="$PWD/VLM V-Lens Final Dataset Complete"
python3 tools/layer_readouts.py read --model Qwen3-VL-8B-Instruct --maps results/fits_matched/maps --out results/fits_matched/readouts --families openimages_orig_auth openimages_edit_auth openimages_orig_eval nuscenes_edit_auth vins_orig_auth vins_edit_auth gemini_orig_auth wmdp_bio_sand wmdp_chem_sand wmdp_cybe_sand
echo EXIT $?
python3 tools/score_matched.py
echo SCORED $?
