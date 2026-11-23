#!/bin/bash
# Copies the figures the deck shows (all made by EP4's main.py, or by EP3's buffon.py) and cuts one rotation out of
# each looped video (the browser loops it).
set -e
cd "$(dirname "$0")"
ROOT=../..
MEASURE=$ROOT/EP4/runs/measure/2026-11-22_20-25-24-Welton-all
SHOW=$ROOT/EP4/runs/make/2026-11-22_20-07-02-1024-f300-gpu32-snr10
LADDER=$ROOT/EP4/runs/make/2026-11-23_04-02-09-ladder-1024-f300-gpu32-n1e09-n1e13
STAR_2K=$ROOT/EP4/runs/make/2026-11-23_02-19-28-2048-f300-gpu32-snr10
EP3=$ROOT/EP3/runs/2026-11-08_20-36-22-gpu-fp64
FFMPEG=${FFMPEG:-$HOME/miniconda3/envs/aga511/bin/ffmpeg}
ONE_ROTATION=300
mkdir -p figures videos
for name in amdahl_ceiling_fp64 costs_large_n registers_and_occupancy phases_fp64 costs_compute costs_memory; do
    cp $MEASURE/figures/$name.png figures/
done
cp $MEASURE/validation/independent-gpu64/comparison.png figures/independent_comparison.png
cp $EP3/amdahl.png figures/ep3_amdahl.png
cp $EP3/error.png figures/ep3_error.png
$FFMPEG -y -loglevel error -i $STAR_2K/observer_mu0071_i1.0411_frame_0.png -q:v 2 figures/star_2k_60deg.jpg
one_rotation() {
    $FFMPEG -y -loglevel error -i "$1" -frames:v $ONE_ROTATION -c copy -an -movflags +faststart "$2"
}
one_rotation $SHOW/observers_around.mp4 videos/observers_around.mp4
one_rotation $SHOW/observers_grid.mp4 videos/observers_grid.mp4
one_rotation $LADDER/snr_ladder.mp4 videos/snr_ladder.mp4
echo "figures: $(du -sh figures | cut -f1), videos: $(du -sh videos | cut -f1)"
