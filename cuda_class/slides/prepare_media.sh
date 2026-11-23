#!/bin/bash
# Copies the figures the slides use and turns the videos into frames for the animate package.
set -e
cd "$(dirname "$0")"
ROOT=../..
MEASURE=$ROOT/EP4/runs/measure/2026-11-22_20-25-24-Welton-all/figures
SHOW=$ROOT/EP4/runs/make/2026-11-22_20-07-02-1024-f300-gpu32-snr10
LADDER=$ROOT/EP4/runs/make/2026-11-22_20-23-46-ladder-256-f300-gpu32-n1e08-n1e11
EP3=$ROOT/EP3/runs/2026-11-08_20-36-22-gpu-fp64
FFMPEG=${FFMPEG:-$HOME/miniconda3/envs/aga511/bin/ffmpeg}
FRAMES=150
mkdir -p figures frames
cp $MEASURE/amdahl_ceiling_fp64.png $MEASURE/costs_large_n.png $MEASURE/registers_and_occupancy.png \
   $MEASURE/strong_fp64.png $MEASURE/speed.png figures/
cp $EP3/amdahl.png figures/ep3_amdahl.png
cp $SHOW/observers_around_frame_0.png figures/around_still.png
cp $SHOW/observers_grid_frame_0.png figures/grid_still.png
cp $LADDER/snr_ladder_frame_0.png figures/ladder_still.png
rm -f frames/*
EVERY_OTHER="select='not(mod(n\,2))'"
$FFMPEG -loglevel error -i $SHOW/observers_around.mp4 -vf "$EVERY_OTHER,scale=-2:540" -fps_mode vfr \
    -frames:v $FRAMES -q:v 5 frames/around-%03d.jpg
$FFMPEG -loglevel error -i $LADDER/snr_ladder.mp4 -vf "$EVERY_OTHER,scale=1000:-2" -fps_mode vfr \
    -frames:v $FRAMES -q:v 5 frames/ladder-%03d.jpg
echo "figures: $(du -sh figures | cut -f1), frames: $(du -sh frames | cut -f1) ($(ls frames | wc -l) files)"
