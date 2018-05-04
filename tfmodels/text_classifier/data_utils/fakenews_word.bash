#!/bin/bash

datapath=../data/fakenews
seqlen=50
wordlen=5
n_classes=2

./prepare_word.bash ${datapath} ${seqlen} ${wordlen} ${n_classes}
