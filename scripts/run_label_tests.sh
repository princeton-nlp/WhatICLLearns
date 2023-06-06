#!/bin/bash

# Get arguments
SEED=${SEED:-"0"}                                           # seed (for slurm jobs, uses slurm job id) 
MODEL=${MODEL:-}                                            # model
VAL_EXAMPLES=${VAL_EXAMPLES:-}                              # number of examples to evaluate on
API_KEY=${API_KEY:-}                                        #   whose openai key to use

DATASET=${DATASET:-"sst2"}                                  # dataset
MODE=${MODE:-"VANILLA"}                                     # experiment mode: VANILLA, RANDOM_LABEL, NUMBER_LABEL, ABSTRACT_LABEL, LETTER_LABEL
INPUT_FORMAT=${INPUT_FORMAT:-}                              # input_format
OUTPUT_FORMAT=${OUTPUT_FORMAT:-"classification"}            # output_format
PROMPT_LEN=${PROMPT_LEN:-8}                                 # dataset
LABEL_SPACE=${LABEL_SPACE:-}
TEMPLATE_NUM=${TEMPLATE_NUM:-}
COMPUTE_SUMMARY_STATS=${COMPUTE_SUMMARY_STATS:-"FALSE"}
INDEX=${INDEX:-}                                            # Job task id

cd ..

# Create scratch directory if it doesn't exist
scratch_dir=$SCRATCH_DIR
if [ ! -d ${scratch_dir} ]; then
    mkdir ${scratch_dir}
fi


# Use Slurm Job ID?
if [ ! -z "$INDEX" ]
then
    SEED=$INDEX
fi


case $MODE in
    VANILLA)
        ADDITIONAL_ARGS="
            --constrained_decoding
            --demo_sep_lines 3
            --natlan_template_number ${TEMPLATE_NUM}
        "
        ;;
    RANDOM_LABEL)
        ADDITIONAL_ARGS="
            --constrained_decoding
            --demo_sep_lines 3
            --natlan_template_number ${TEMPLATE_NUM}
            --random_label
        "
        ;;
    NUMBER_LABEL)
        ADDITIONAL_ARGS="
            --constrained_decoding
            --demo_sep_lines 3
            --minimal_template_number ${TEMPLATE_NUM}
            --label_space number
        "
        ;;
    LETTER_LABEL)
        ADDITIONAL_ARGS="
            --constrained_decoding
            --demo_sep_lines 3
            --minimal_template_number ${TEMPLATE_NUM}
            --label_space letter
        "
        ;;
    ABSTRACT_LABEL)
        ADDITIONAL_ARGS="
            --constrained_decoding
            --demo_sep_lines 3
            --minimal_template_number ${TEMPLATE_NUM}
            --label_space abstract
        "
        ;;

    
    *)
        ADDITIONAL_ARGS="
        "
        ;;
esac

echo "---------------------------------------------------"
echo "Model: ${MODEL}"
echo "Mode ${MODE}: ${DATASET}, Greedy Decoding"
echo "$PROMPT_LEN demos, $VAL_EXAMPLES examples total"
echo "Input Format $INPUT_FORMAT, Output Format $OUTPUT_FORMAT"
echo "Additional Args: $ADDITIONAL_ARGS"
echo "Template Args: $TEMPLATE_ARGS"
echo "Extra Args: $@"

echo $COMPUTE_SUMMARY_STATS 
if [ "$COMPUTE_SUMMARY_STATS" = "TRUE" ];
then
    echo "Computing summary stats"
    python -m spb.compute_summary_stats -c config.ini ${MODEL}_icl \
        --num_prompt_ex $PROMPT_LEN \
        --datasets $DATASET \
        --val_examples $VAL_EXAMPLES \
        --episodes $SEED \
        --output_format $OUTPUT_FORMAT \
        --api_key_name $API_KEY \
        --dc_template $TEMPLATE_NUM \
        --dc_mode $MODE \
        $ADDITIONAL_ARGS \
        $@
else
    python -m spb.main -c config.ini ${MODEL}_icl \
        --num_prompt_ex $PROMPT_LEN \
        --datasets $DATASET \
        --val_examples $VAL_EXAMPLES \
        --episodes $SEED \
        --output_format $OUTPUT_FORMAT \
        --api_key_name $API_KEY \
        $ADDITIONAL_ARGS \
        $@
fi

