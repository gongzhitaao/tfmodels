#!/bin/bash

seqlen=50
wordlen=5
data=fakenews
n_classes=2

cd .. && \
python run_charlstm.py \
       --batch_size 20 \
       --data data/${data}/char-seqlen-${seqlen}-wordlen-${wordlen}.npz \
       --drop_rate 0.2 \
       --embedding_dim 128 \
       --epochs 5 \
       --feature_maps 25 50 75 100 125 150 \
       --highways 1 \
       --kernel_size 1 2 3 4 5 6 \
       --lstm_units 256 \
       --lstms 2 \
       --n_classes ${n_classes} \
       --name ${data}-char-sigm-seqlen-${seqlen}-wordlen-${wordlen} \
       --seqlen ${seqlen} \
       --unipolar \
       --vocab_size 128 \
       --wordlen ${wordlen}
