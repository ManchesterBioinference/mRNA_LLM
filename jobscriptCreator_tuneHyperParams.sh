#!/bin/bash
# This script will be used to run pieces of the pipeline in parallel when testing the effect of amount of training data on performance for the three models (SVM, iDeep, and 3UTRBERT-fly)

learning_rates=(1e-5 2e-5 3e-5 4e-5 5e-5) #(3e-5 3.5e-5 2.5e-5 4e-5 2e-5 4.5e-5 1.5e-5 5e-5 1e-5)
nepochs=(32 35 38 41 44 47 50 53 56 59 62 65) #(50 53 47 56 44 59 41 62 38 65 35 68 32 71 29) #(40 39 41 38 42 37 43 36 44 35 45 34 46)
warm_steps=(0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15)


# Remove all files in the jobscriptStatus dir
mkdir -p jobscriptStatus
#rm -f jobscriptStatus/*

for n in "${nepochs[@]}"; do
    sed -i -e '0,/num_train_epochs: [0-9]*/b' -e "s/num_train_epochs: [0-9]*/num_train_epochs: ${n}/" params.yaml
    for lr in "${learning_rates[@]}"; do
        sed -i -e '0,/learning_rate: [0-9e-]*/b' -e "s/learning_rate: [0-9e-]*/learning_rate: ${lr}/" params.yaml
        for warm in "${warm_steps[@]}"; do
            if [[ -f "jobscriptStatus/lr${lr}_n${n}_warm${warm}" ]]; then
                continue
            fi
            #sed -i "s/warmup_steps: [0-9]*/warmup_steps: ${warm}/" params.yaml
            sed -i -e '0,/warmup_percent: [0-9\.]*/b' -e "s/warmup_percent: [0-9\.]*/warmup_percent: ${warm}/" params.yaml
            echo "Running learning rate ${lr}, num epochs ${n}, and warm steps ${warm}..."
            qsub jobscriptSubcommand_gpu_tuneHyperParams.sh $lr $n $warm

            while [[ ! -f "jobscriptStatus/lr${lr}_n${n}_warm${warm}" ]]; do
                sleep 2
            done
        done
    done
done