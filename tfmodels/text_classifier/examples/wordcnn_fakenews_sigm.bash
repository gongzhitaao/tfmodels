#!/bin/bash

seqlen=50
data=fakenews
n_classes=2

cd .. && \
python run_wordcnn.py \
       --batch_size 128 \
       --data data/${data}/word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --epochs 5 \
       --filters 128 \
       --kernel_size 3 \
       --n_classes 2 \
       --name ${data}-word-sigm-seqlen-${seqlen} \
       --seqlen ${seqlen} \
       --unipolar \
       --units 128
