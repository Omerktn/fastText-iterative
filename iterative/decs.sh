#!/bin/bash
input_data="./data/enwik8"
epoch=30
thread=8
version="_v2"
dim_array=()
acc_array=()

start_dim=200
dim=180
former_dim=$dim

first_output=./models/decwik_e${epoch}_d${start_dim}t${dim}${version}
./fasttext skipgram -input ${input_data} -output ${first_output} \
           -pretrainedVectors ./models/enwik_d${start_dim}_e${epoch}.vec -decVectors ${dim} \
           -thread ${thread} -epoch ${epoch} -dim ${start_dim}

analogy_out=$(./fasttext test-analogies ${first_output}.bin ~/Documents/analogy_test)

dim_array+=($dim)
re="accuracy: ([0-1].[0-9]*)"
if [[ $analogy_out =~ $re ]]; then
    match=${BASH_REMATCH[1]}
    acc_array+=($match)
fi

for dim in 160 140
do
    new_output=./models/decwik_e${epoch}_d${start_dim}t${dim}${version}
    pretrained=./models/decwik_e${epoch}_d${start_dim}t${former_dim}${version}.vec

    ./fasttext skipgram -input ${input_data} -output ${new_output}  \
               -pretrainedVectors ${pretrained} -decVectors ${dim} \
               -thread ${thread} -epoch ${epoch} -dim ${former_dim}
    former_dim=$dim

    analogy_out=$(./fasttext test-analogies ${new_output}.bin ~/Documents/analogy_test)

    if [[ $analogy_out =~ $re ]]; then
        match=${BASH_REMATCH[1]}
        acc_array+=($match)
        echo "Catched :" $match "at dim " $dim
    fi
    dim_array+=($dim)
done

# Print lists

printf "dims_ = ["
for value in "${dim_array[@]}"
do
    printf $value
    printf ", "
done
printf "]\naccs_ = ["
for value in "${acc_array[@]}"
do
    printf $value
    printf ", "
done
printf "]\n"

