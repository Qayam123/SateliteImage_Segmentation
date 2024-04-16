#!/usr/bin/bash
conda activate samenv
mkdir outputs
python3 satelite.py
python3 segment.py
