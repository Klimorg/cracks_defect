#!/bin/bash

create_dataset(){
    mkdir ./datas/raw_dataset
    for f in ./datas/raw_datas/*; do
        if [ -d "$f" ]; then
            # $f is a directory
            b=$(basename $f)
            echo "Making new directories for" $b
            mkdir ./datas/raw_dataset/$b
            #ls $f/ | head -$1
            echo "Copying the first $1 pictures for folder $b"
            for F in $(ls $f/ | sort | head -$1); do
                cp $f/$F ./datas/raw_dataset/$b/$F
            done
        fi
    done
}

create_dataset $1
#mv `ls Positive/ | head -10` ../small_dataset/Positive/