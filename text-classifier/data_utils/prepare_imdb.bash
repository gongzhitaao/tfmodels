#!/bin/bash

prefix=../data/imdb
name=imdb
seqlen=50
wordlen=5
n_classes=2

./prepare_data.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
