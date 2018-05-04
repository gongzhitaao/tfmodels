#!/bin/bash

# Assume that the raw text files are stored in ${datapath}.  The train and test
# data files for each class is named train-0.txt, train-1.txt, ...,
# train-(${n_classes}-1), test-0.txt, test-1.txt, ..., test-(${n_classes}-1)

datapath=$1                     # base name for the output
seqlen=$2                       # maximum sequence length
wordlen=$3                      # maximum word length
n_classes=$4                    # number of categories

labels=$(seq 0 $((${n_classes} - 1)))

# Assume that the raw file names are in the format xxxx-label.txt, e.g.,
# imdb-0.txt, imdb-1.txt.
raw_file="${datapath}/%s-%d.txt"

# The raw file is tokenized first and the tokens are stored in this file.  The
# tokenize operation is expensive, so we do not re-tokenize if the token files
# already exists.
token_file="${datapath}/tmp/%s-%d-tokens.txt"

# The tokenized file is padded at character-level and store result in this file.
charpad="${datapath}/tmp/%s-%d-charpad.txt"

# The data samples of different classes are gathers together for train and test,
# respectively.
charall="${datapath}/tmp/%s-charall.txt"

# Finally the data are converted to indices and stored in one npz file
data="${datapath}/char-seqlen-%d-wordlen-%d.npz"

function tokenize {
    for i in train test; do
        for j in ${labels}; do
            f0=$(printf ${raw_file} ${i} ${j})
            f1=$(printf ${token_file} ${i} ${j})
            [[ ! -f "${f1}" ]] && \
                python -u 0_tokenize.py \
                       --unescape --cleanup \
                       ${f0} > ${f1}
            echo "Wrote ${f1}"
        done
    done
}

function charpad {
    for i in train test; do
        for j in ${labels}; do
            f0=$(printf ${token_file} ${i} ${j})
            f1=$(printf ${charpad} ${i} ${j})
            python -u 1_charpad.py \
                   --seqlen ${seqlen} --wordlen ${wordlen} \
                   --ascii --encode \
                   --sow '{' --eow '}' --eos '+' --pad ' ' --unk '|' \
                   ${f0} > ${f1}
            echo "Wrote ${f1}"
        done
    done
}

function charmerge {
    for i in train test; do
        f1=$(printf ${charall} ${i})
        [ -f ${f1} ] && mv ${f1}{,.bak}
        for j in ${labels}; do
            f0=$(printf ${charpad} ${i} ${j})
            sed -e "s/^/${j} /" \
                ${f0} >> ${f1}
        done
    done
}

function char2index {
    python 2_char2index.py \
           --train $(printf ${charall} "train") \
           --test $(printf ${charall} "test") \
           --output $(printf ${data} ${seqlen} ${wordlen})
}

mkdir -p ${datapath}/tmp
tokenize
charpad
charmerge
char2index
