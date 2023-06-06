#!/bin/bash

##########################################################
# Testing for ICL paper.
##########################################################

DO_TEST="./run_label_tests.sh"

VAL_EXAMPLES=1
SUMMARY_STATS=FALSE
API_KEY=xyz # Make an environment variable named xyz_openai which holds your OpenAI API key.
TEMPLATE_NUM=1

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


for TASK in ${TASKS[@]}; do
    for MODEL in gpt3_ada opt_125m; do
        for MODE in ABSTRACT_LABEL RANDOM_LABEL NUMBER_LABEL ABSTRACT_LABEL LETTER_LABEL; do
            MODEL=$MODEL \
            VAL_EXAMPLES=1 \
            MODE=$MODE \
            DATASET=$TASK \
            INDEX="0" \
            TEMPLATE_NUM=1 \
            PROMPT_LEN=8 \
            API_KEY=danqi \
            SCRATCH_DIR="/n/fs/scratch/nlp-jp7224/"\
            COMPUTE_SUMMARY_STATS="FALSE" \
            bash ./run_label_tests.sh \
                --output_dir test_output \
                $@
        done;
    done;
done;
