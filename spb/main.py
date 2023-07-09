import argparse
import configparser
import json
import logging
import os
import time
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, GPT2Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from spb.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from spb.bench_datasets import load_dataset
from spb.evaluate import evaluate_results, evaluate_icl
from spb.utils import get_episode_indices


OPENAI_MODELS_LIST = [
    "gpt3-ada", 
    "gpt3-babbage",  
    "gpt3-curie", 
    "gpt3-davinci", 
    "text-davinci-002", 
    "code-davinci-002"
]

LLAMA_MODELS_LIST = [
    "/scratch/gpfs/jp7224/llama/7B",
    "/scratch/gpfs/jp7224/llama/13B",
    "/scratch/gpfs/jp7224/llama/30B",
    "/scratch/gpfs/jp7224/llama/65B",
]

OPT_MODELS_LIST = [
    'facebook/opt-13b',
    'facebook/opt-125m',
    'facebook/opt-30b',
    'facebook/opt-350m',
    'facebook/opt-66b',
    'opt-66b',
    'facebook/opt-2.7b',
    'facebook/opt-6.7b',
]

CAUSAL_LM_LIST = OPT_MODELS_LIST + LLAMA_MODELS_LIST

def set_data_args_defaults(data_args):
    """
    Sets default values for various sequence/chunk lengths.
    """

    # process arguments related to max length
    if data_args.max_output_seq_length_eval is None:
        # defaults first to max_output_seq_length, then max_seq_length_eval, then max_seq_length
        data_args.max_output_seq_length_eval = data_args.max_output_seq_length \
                                               or data_args.max_seq_length_eval \
                                               or data_args.max_seq_length

    if data_args.max_output_seq_length is None:
        # defaults to max_seq_length
        data_args.max_output_seq_length = data_args.max_seq_length

    if data_args.max_seq_length_eval is None:
        # defaults to max_seq_length
        data_args.max_seq_length_eval = data_args.max_seq_length
       
    return data_args

def get_split(training_args):
    """ 
    Should we train on dev, test, or both? 
    """

    if training_args.do_eval:
        return 'dev'
    if training_args.do_predict:
        return 'test'
    
def get_tokenizer(model_args):
    """
    Get tokenizer for model.
    """

    # tokenizer for GPT-3 models
    if model_args.model_name_or_path in OPENAI_MODELS_LIST:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        tokenizer.pad_token = tokenizer.eos_token
        logging.warning(f"Using GPT2 Tokenizer with pad token {tokenizer.pad_token}")

    # tokenizer for OPT models
    elif model_args.model_name_or_path in OPT_MODELS_LIST:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # tokenizer for LLAMA models
    elif model_args.model_name_or_path in LLAMA_MODELS_LIST:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        )

        # If GPT-Neo, GPT-J, GPT-2, etc., don't use eos token for pad -- causes some issues with generation
        if model_args.model_name_or_path in CAUSAL_LM_LIST:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        
    return tokenizer

def get_output_dir(args, model_args, data_args, training_args):
    """
    Get unique output_dir name
    """

    output_dir = os.path.join(
        training_args.output_dir,
        f'{args.job}'
        f'-{model_args.model_name_or_path.split("/")[-1]}'
        f'-len{data_args.max_seq_length}'
    )

    output_dir += f'-b{training_args.per_device_train_batch_size}' \
                  f'-{data_args.train_split}'

    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'

    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'

    if data_args.train_examples != -1:
        output_dir += f'-size{data_args.train_examples:.2f}' + f'-dataset_{data_args.datasets}'

    if training_args.icl_only:
        output_dir += '-icl' + f'-dataset_{data_args.datasets}' + f'-prompt_len{data_args.num_prompt_ex}'

    if data_args.input_seq_length != -1:
        output_dir += f"-input_seq_length{data_args.input_seq_length}"

    if data_args.label_space is not None:
        output_dir += f"-label_space{data_args.label_space}"
    
    if data_args.constrained_decoding:
        output_dir += "-constrained_decoding"
    
    if data_args.random_label:
        output_dir += "-random_label"

    if data_args.demo_sep_lines != 1:
        output_dir += f"-demo_sep_lines{data_args.demo_sep_lines}"
        
    if data_args.natlan_template_number is not None:
        output_dir += f"-natlan_template_number{data_args.natlan_template_number}"
        
    if data_args.minimal_template_number is not None:
        output_dir += f"-minimal_template_number{data_args.minimal_template_number}"

    return output_dir


def get_model(model_args, config, train_phase=True, model_dir=None):
    """
    Returns either pre-trained model for zero-shot/training or locally stored fine-tuned model for evaluation.
    """

    # GPT-3/Codex: we only need to access OpenAI API
    if model_args.model_name_or_path in OPENAI_MODELS_LIST:
        model = model_args.model_name_or_path
    
    # OPT-13B
    elif model_args.model_name_or_path in OPT_MODELS_LIST:
        # preparing memory/device map for accelerate
        last_time = time.time()
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-2}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}

        model_weights_path = model_args.model_weights_dir + model_args.model_name_or_path
        
        if os.path.exists(model_weights_path):
            offload_path = model_args.model_offload_dir + model_args.model_name_or_path
            logging.warning(f"Loading weights from {model_weights_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_weights_path,
                device_map='auto',
                torch_dtype=torch.float16,
                max_memory=max_memory, 
                offload_folder=offload_path,
            )

        else:
            logging.warning(f"Loading weights from {model_args.model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.float16,
                cache_dir=model_weights_path,
                offload_folder=offload_path,
            )
            model.save_pretrained(model_weights_path)
            model.cuda()

        
        print("CUDA Loading Time: {} min".format(round(time.time()-last_time)/60))
        model.config.pad_token_id = model.config.eos_token_id
        
    elif model_args.model_name_or_path in LLAMA_MODELS_LIST:
        last_time = time.time()
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-2}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        offload_path = model_args.model_offload_dir + model_args.model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    device_map='auto',
                    torch_dtype=torch.float16,
                    max_memory=max_memory, 
                    offload_folder=offload_path,
            )
            
    elif model_args.model_name_or_path in LLAMA_MODELS_LIST:
        config = LlamaConfig()
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map='auto',
        )

    # seq2seq models, e.g. T5, BART, etc.
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            force_download=True
        )

        model.cuda()
    return model 

def get_model_config(model_args):
    """
        Gets model config.
    """

    # GPT-3/Codex: no model config needed
    if model_args.model_name_or_path in OPENAI_MODELS_LIST:
        config = None

    elif model_args.model_name_or_path in LLAMA_MODELS_LIST:
        config = LlamaConfig.from_pretrained(
                model_args.model_name_or_path if model_args.config_name else model_args.model_name_or_path, 
                cache_dir=model_args.cache_dir,
        )
    
    else:
        config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path, 
                cache_dir=model_args.cache_dir,
        )

    return config

def get_evaluation_output_filename(data_args):
    """
    Construct file name for the evaluation results
    """

    evaluation_output_filename = 'results'
    if data_args.num_beams is not None:
        evaluation_output_filename += f'-{data_args.num_beams}beams'
    if data_args.max_seq_length_eval is not None:
        evaluation_output_filename += f'-len{data_args.max_seq_length_eval}'
    if data_args.datasets is not None:
        evaluation_output_filename += f'-len{data_args.max_seq_length_eval}'

    return evaluation_output_filename

def main(args, model_args, data_args, training_args, model, tokenizer, model_dir, ep_idx):
    dataset_list = data_args.datasets.split(",")
    

    for dataset_name in dataset_list:
        # If doing ICL, training, or collecting dataset attribute information (for GitHub README), get train dataset
        if training_args.icl_only or training_args.do_train or data_args.save_dataset_stats:
            logging.warning(f'Process dataset {dataset_name} (train)')
            train_dataset = load_dataset(
                            dataset_name, 
                            data_args, 
                            causal_lm=model_args.model_name_or_path in CAUSAL_LM_LIST,
                            max_input_length=data_args.max_seq_length, 
                            max_output_length=data_args.max_output_seq_length,
                            num_examples=data_args.train_examples,
                            seed=ep_idx,
                            split=data_args.train_split,
                            tokenizer=tokenizer, 
                        )

            logging.warning(f'Getting ICL examples for {dataset_name} (train)')
            

        # Prepare for evaluation dataset
        split = get_split(training_args)
        logging.warning(f'Evaluate on {dataset_name} {split}')
        results_fn = None
        evaluation_output_filename = get_evaluation_output_filename(data_args)
        
        # Performing in-context learning:
        if training_args.icl_only and not training_args.evaluate_results_only:
            logging.warning("Now performing in-context learning")

            response_dict = evaluate_icl(
                model=model,
                train_dataset=train_dataset,
                dataset_name=dataset_name,
                data_args=data_args,
                tokenizer=tokenizer,
                split="dev" if dataset_name != "ade" else "test",
                seed=ep_idx,
                subset=None,
                gpu=args.gpu,
                causal_lm=model_args.model_name_or_path in CAUSAL_LM_LIST,
            )

                
            response_df = pd.DataFrame(response_dict)
            results_fn = os.path.join(model_dir, evaluation_output_filename + f'-{dataset_name}-{split}-output.csv')
            logging.warning(f"Saving results to {results_fn}")
            response_df.to_csv(results_fn, sep="\t")

        # # Run analysis on saved results
        if training_args.evaluate_results or training_args.evaluate_results_only:
            if results_fn is None and data_args.results_fn is None:
                logging.warning("Using previously stored results")
                results_fn = os.path.join(model_dir, evaluation_output_filename + f'-{dataset_name}-{split}-output.csv')
            
            elif results_fn is None:
                results_fn = data_args.results_fn

            res = evaluate_results(
                    results_fn=results_fn,
                    model=model, 
                    model_name=model_args.model_name_or_path,
                    dataset_name=dataset_name, 
                    data_args=data_args, 
                    tokenizer=tokenizer, 
                    split="dev" if dataset_name != "ade" else "test", 
                    seed=ep_idx, 
                    batch_size=training_args.per_device_eval_batch_size, 
                    gpu=args.gpu, 
                    causal_lm=model_args.model_name_or_path in CAUSAL_LM_LIST,
                    subset=None
                )
  
            with open(
                os.path.join(model_dir, evaluation_output_filename + f'-{dataset_name}-{split}.json'), 'w'
            ) as f:
                json.dump(res, f, indent=0)



if __name__ == "__main__":
    # assert torch.cuda.is_available(), 'CUDA not available'
    print('__Number CUDA Devices:', torch.cuda.device_count())
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', 
                        help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true', default=False, 
                        help='run evaluation only')
    parser.add_argument('--evaluate_checkpoints', action='store_true', default=False,
                        help='evaluate intermediate checkpoints instead of the final model')
    parser.add_argument('--evaluate_last_checkpoint', action='store_true', default=False,
                        help='evaluate the last intermediate checkpoint instead of the final model')
    parser.add_argument('--evaluate_checkpoint_in_dir', type=str, default=None,
                        help='evaluate the checkpoint in the given directory')
    parser.add_argument('-a', '--evaluate_all', action='store_true', default=False,
                        help='evaluate intermediate checkpoints together with the final model')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use for evaluation')
    parser.add_argument('-v', '--verbose_results', action='store_true', default=False,
                        help='print results for each evaluation run')

    args, remaining_args = parser.parse_known_args()

    # read config file
    config = configparser.ConfigParser(allow_no_value=False, interpolation=None)
    config.read(args.config_file)
    job = args.job
    assert job in config

    # set defaults for other arguments
    defaults = {
        'overwrite_output_dir': True,
        'overwrite_cache': True,
        'per_device_eval_batch_size': 4,
        'learning_rate': 5e-4,
        'logging_steps': 1,     # do not log by default
        'save_steps': 0,        # do not save checkpoints by default
    }

    # the config file gives default values for the command line arguments
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            # interpret True/False as boolean
            defaults[key] = config.getboolean(job, key)
        if defaults[key] == 'None':
            # interpret as None
            defaults[key] = None

    if args.eval:
        # run evaluation only
        defaults['do_train'] = False

    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)

    data_args = set_data_args_defaults(data_args)
    model_config = get_model_config(model_args)
    model = get_model(model_args, model_config, train_phase=True, model_dir=None)
    tokenizer = get_tokenizer(model_args)
    output_dir = get_output_dir(args, model_args, data_args, training_args)

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    # setup logging
    logging.basicConfig(
      filename=os.path.join(output_dir, 'logs.log'),
      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      level=logging.WARNING,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # construct list of episode indices
    episode_indices = get_episode_indices(data_args.episodes)
    
    results = []
    split = get_split(training_args)
    evaluation_output_filename = get_evaluation_output_filename(data_args)
    for ep_idx in episode_indices:
        logging.warning(f'Episode {ep_idx} of {len(episode_indices)-1}')

        # make episode output directory
        episode_output_dir = os.path.join(output_dir, f'ep{ep_idx}')
        training_args.output_dir = episode_output_dir
        logging.warning(f'Output directory: {episode_output_dir}')

        try:
            os.mkdir(episode_output_dir)
        except FileExistsError:
            pass

        # perform training/evaluation
        results.append(main(args, model_args, data_args, training_args, model, tokenizer, episode_output_dir, ep_idx))

    logging.warning(f'Model weights and intermediate checkpoints saved in {output_dir}')