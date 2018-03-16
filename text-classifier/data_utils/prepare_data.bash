#!/bin/bash

prefix=$1
name=$2
seqlen=$3
wordlen=$4
n=$5

labels=$(seq 0 $((${n} - 1)))

function tokenize {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}
            python -u 0_tokenize.py \
                   --unescape --cleanup \
                   ${prefix}/${fn}.txt > \
                   ${prefix}/${fn}-tokens.txt
        done
    done
}

function charpad {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}
            python -u 1_charpad.py \
                   --seqlen ${seqlen} --wordlen ${wordlen} \
                   --ascii --encode \
                   --sow '{' --eow '}' --eos '+' --pad ' ' --unk '|' \
                   ${prefix}/${fn}-tokens.txt > \
                   ${prefix}/${fn}-seqlen-${seqlen}-wordlen-${wordlen}.txt
        done
    done
}

function charmerge {
    for pre in train test; do
        [ -f ${prefix}/${pre}-char.txt ] && \
            cp ${prefix}/${pre}-char.txt{,.bak}
        for lab in ${labels}; do
            fn=${prefix}/${pre}-${lab}-seqlen-${seqlen}-wordlen-${wordlen}.txt
            sed -e "s/^/${lab} /" \
                ${fn} >> ${prefix}/${pre}-char.txt
        done
    done
}

function char2index {
    python 2_char2index.py \
           --train ${prefix}/train-char.txt --test ${prefix}/test-char.txt \
           --output ${prefix}/${name}-char-seqlen-${seqlen}-wordlen-${wordlen}.npz
}

function wordpad {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}
            python -u 1_wordpad.py \
                   --seqlen ${seqlen} \
                   --pad '<pad>' --eos '<eos>' --unk '<unk>' \
                   ${prefix}/${fn}-tokens.txt > \
                   ${prefix}/${fn}-seqlen-${seqlen}.txt
        done
    done
}

function wordmerge {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}-seqlen-${seqlen}
            sed -e "s/^/${lab} /" ${prefix}/${fn}.txt > \
                ${prefix}/${pre}-word.txt
        done
    done
}

function word2index {
    python 2_token2index.py \
           --w2v ~/data/glove/glove.840B.300d.w2v \
           --train ${prefix}/train-word.txt --test ${prefix}/test-word.txt \
           --output ${prefix}/${name}-word-seqlen-${seqlen}.npz
}

tokenize

charpad
charmerge
char2index

# wordpad
# wordmerge
# word2index
