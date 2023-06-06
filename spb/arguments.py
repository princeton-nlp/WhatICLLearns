from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """

    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the results will be written."}
    )
    
    zero_shot: bool = field(
        default=False,
        metadata={"help": "Zero-shot setting"}
    )

    """
    SPB arguments
    """

    report_to: str = "wandb"

    evaluate_results: bool = field(
        default=True,
        metadata={"help": "Run error analysis on outputted results, after icl or ft"}
    )

    evaluate_results_only: bool = field(
        default=False,
        metadata={"help": "Run error analysis on outputted results that have been saved to a file"}
    )
    
    icl_only: bool = field(
        default=False
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models"}
    )

    """
    SPB arguments
    """

    model_weights_dir: str = field(
        default=None,  metadata={"help": "Pretrained weights folder"}
    )

    model_offload_dir: str = field(
        default=None,  metadata={"help": "Offloaded model weights folder"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )

    eval_datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names. Defaults to the train datasets."}
    )

    train_split: str = field(
        default='train',
        metadata={"help": "The datasplit for training. Can be 'train', 'dev', 'test', etc."}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, shorter sequences will be padded."
        },
    )

    max_output_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length (default is the same as input)"
        },
    )

    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    overwrite_hf_data: bool = field(
        default=False, metadata={"help": "Overwrite saved data from Huggingface"}
    )

    overwrite_summary_stats: bool = field(
        default=False, metadata={"help": "Overwrite saved data from summary stat analysis"}
    )

    episodes: str = field(
        default='0', metadata={"help": "Episode indices -- a single number such as 3 or an interval such as 1-4\n"
                                       "The index is also used as random seeds and this setting is therefore used to "
                                       "repeat multiple experiments."}
    )

    num_beams: int = field(
        default=None,
        metadata={"help": "Number of beams for beam search during generation (only affects evaluation)"}
    )

    max_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "Maximum input sequence length at evaluation time (default is equal to max_seq_length)"
        },
    )

    max_output_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length at evaluation time (default is the same as input)"
        },
    )
    
    input_format: str = field(
        default=None,
        metadata={"help": "Input format"}
    )
    
    output_format: str = field(
        default=None, metadata={"help": "Output format"}
    )
    
    """
    SPB arguments
    """
    demo_sep_lines: int = field(
        default = 3, metadata={"help": "How many newlines to separate demos"}
    )
    
    minimal_template_number: str = field(
        default = None, metadata={"help": "Template number for minimal templates"}
    )
    
    natlan_template_number: str = field(
        default = None, metadata={"help": "Template number for natural language templates"}
    )
    
    max_dataset_size: int = field(
        default = 500, metadata={"help": "Max number of examples to have in dataset"}
    )
    
    random_label: bool = field(
        default = False, metadata={"help": "Use random labels"}
    )

    batch_size: int = field(
        default = 1, metadata={"help": "Batch size for generation."}
    )

    max_time_limit: int = field(
        default = None, metadata={"help": "Length of job."}
    )

    do_sample: bool = field(
        default = False, metadata={"help": "Use sampling instead of greedy decoding."}
    )


    max_new_tokens: int = field(
        default = 128, metadata={"help": "Max new tokens to generate."}
    )

    api_key_name: str = field(
        default = None, metadata={"help": "Name of environment variable holding OpenAI key"}
    )
    
    replace_labels: str = field(
        default = None, metadata={"help": "Perturb labels. Options: perturb, random"}
    )
    
    label_space: str = field(
        default = None, metadata={"help": "Use different set of labels. Options: number, lmbff, unnatural, abstract"}
    )

    train_examples: int = field(
        default = -1, metadata={"help": "Number of training examples"}
    )

    val_examples: int = field(
        default = 25, metadata={"help": "Number of validation examples"}
    )

    input_seq_length: int = field(
        default = -1, metadata={"help": "Number of validation examples"}
    )

    results_fn: str = field(
        default=None,
        metadata={"help": "For evaluating results, the location of the file"}
    )
    
    num_sampled_responses: int = field(
        default=1,
        metadata={"help": "For sampled output, how many times to sample"}
    )
    
    num_prompt_ex: int = field(
        default=8,
        metadata={"help": "For sampled output, how many examples to put in prompt"}
    )

    constrained_decoding: bool = field(
        default=False
    )

    # dc_template: str = field(
    #     default = None, metadata={"help": "Template describer"}
    # )
    
    # dc_mode: str = field(
    #     default = None, metadata={"help": "Template describer"}
    # )
    
    # dc_metrics: str = field(
    #     default = None, metadata={"help": "Template describer"}
    # )
    
    # dc_filename: str = field(
    #     default = None, metadata={"help": "Template describer"}
    # )
    
   

    