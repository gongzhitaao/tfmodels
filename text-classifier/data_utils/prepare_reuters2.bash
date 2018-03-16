#!/bin/bash

prefix=../data/reuters/reuters2
name=reuters2
seqlen=50
wordlen=5
n_classes=2

./prepare_data.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
