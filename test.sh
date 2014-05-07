#!/bin/bash

python wsd.py > out
python evaluate.py > results
echo 'Incorrect:'
grep 'No' results | wc -l
echo 'Total:'
wc -l results
