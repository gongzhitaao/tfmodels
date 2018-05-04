#!/bin/bash
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

# The tokenized file is padded and store result in this file.
wordpad="${datapath}/tmp/%s-%d-wordpad.txt"

# The data samples of different classes are gathers together for train and test,
# respectively.
wordall="${datapath}/tmp/%s-wordall.txt"

# Finally the data are converted to indices and stored in one npz file
data="${datapath}/word-seqlen-%d.npz"

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

function wordpad {
    for i in train test; do
        for j in ${labels}; do
            f0=$(printf ${token_file} ${i} ${j})
            f1=$(printf ${wordpad} ${i} ${j})
            python -u 1_wordpad.py \
                   --seqlen ${seqlen} \
                   --pad '<pad>' --eos '<eos>' --unk '<unk>' \
                   ${f0} > ${f1}
            echo "Wrote ${f1}"
        done
    done
}

function wordmerge {
    for i in train test; do
        f1=$(printf ${wordall} ${i})
        [ -f ${f1} ] && mv ${f1}{,.bak}
        for j in ${labels}; do
            f0=$(printf ${wordpad} ${i} ${j})
            sed -e "s/^/${j} /" ${f0} >> ${f1}
        done
    done
}

function word2index {
    python 2_token2index.py \
           --w2v ~/data/glove/glove.840B.300d.w2v \
           --train $(printf ${wordall} "train") \
           --test $(printf ${wordall} "test") \
           --output $(printf ${data} ${seqlen})
}

mkdir -p ${datapath}/tmp
tokenize
wordpad
wordmerge
word2index
