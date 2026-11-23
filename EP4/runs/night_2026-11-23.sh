#!/bin/bash
# The night of 2026-11-23: the 2k image (the same command as the A30's) and a 1k SNR ladder up to 10^13.
cd /home/welto/aga511/EP4
PYTHON=~/miniconda3/envs/aga511/bin/python
LOG=runs/night_2026-11-23.log
echo "start $(date)" > $LOG
$PYTHON main.py make --size 2048 --snr 10 --observers 1.5708,1.0472,0.5236,0 --yes >> $LOG 2>&1
echo "2k done $(date), exit $?" >> $LOG
$PYTHON main.py make --ladder 1e9:1e13 --size 1024 --yes >> $LOG 2>&1
echo "ladder done $(date), exit $?" >> $LOG
