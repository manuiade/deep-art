#!/bin/sh

echo Start downloading cfg folder...

python3 download.py 1AetqzDa30b8GqApzl8g24LFAEaNxTQuT cfg.tar.gz

tar -xf cfg.tar.gz
rm cfg.tar.gz

echo Start downloading data folder...

python3 download.py 1UvJbTCHc2qx-33AM9QEGxcFkphDP0Cee data.tar.gz

tar -xf data.tar.gz
rm data.tar.gz

echo Done.
