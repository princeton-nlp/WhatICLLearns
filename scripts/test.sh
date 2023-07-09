#!/bin/bash

##########################################################
# Testing for ICL paper.
##########################################################

VAL_EXAMPLES=1
API_KEY=xyz # Make an environment variable named xyz_openai which holds your OpenAI API key.
TEMPLATE_NUM=1 # Demonstration template number (1, 2, 3)
MODEL_WEIGHTS_DIR= # location of stored model weights for OPT/LLaMa

TASKS=(
    tweet_eval_hate
    tweet_eval_atheism
    tweet_eval_feminist
    sick 
    financial_phrasebank
    ethos_race
    ethos_gender
    ethos_religion
    ethos_national_origin
    snli 
    sst2 
    trec 
    wnli
    mrpc
    poem
    tweet_eval_emotion
)

MODES=(
    LETTER_LABEL
    ABSTRACT_LABEL
    RANDOM_LABEL
    NUMBER_LABEL 
    VANILLA
)

KS=(
    8
    16
    32
)

for K in ${KS}; do
    for TASK in ${TASKS[@]}; do
        for MODEL in gpt3_ada opt_125m; do
            for MODE in ${MODES[@]}; do
                MODEL=$MODEL \
                VAL_EXAMPLES=$VAL_EXAMPLES \
                MODE=$MODE \
                DATASET=$TASK \
                INDEX="0" \
                TEMPLATE_NUM=$TEMPLATE_NUM \
                PROMPT_LEN=$K \
                API_KEY=$API_KEY \
                MODEL_WEIGHTS_DIR=$MODEL_WEIGHTS_DIR \
                COMPUTE_SUMMARY_STATS="FALSE" \
                bash ./run_label_tests.sh \
                    --output_dir test_output \
                    $@
            done;
        done;
    done;
done;
