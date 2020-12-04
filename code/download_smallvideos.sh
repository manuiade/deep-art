#!/bin/sh

echo Start downloading small video folder...
python3 download.py 186v-K87b1apThbc7r1ja4iEZZmZXWyxY small.tar.gz

tar -xf small.tar.gz
rm small.tar.gz

echo Done.
