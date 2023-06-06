from collections import defaultdict
from re import I
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, generation_utils, Trainer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import argparse
import configparser
import imp
import itertools
import json
import logging
import numpy as np
import os
import time
import pandas as pd
import sys
import torch
import types
import wandb

from spb.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from spb.evaluate import get_avg_results, print_results
from spb.utils import get_device_map
from spb.main import set_data_args_defaults, get_output_dir, get_split, get_evaluation_output_filename, get_episode_indices

def get_model_family(model):
    if "gpt3" in model.lower():
        return "gpt3"
    elif "opt" in model.lower():
        return "opt"
    else:
        assert False, "No recognized model"

def main():
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
    output_dir = get_output_dir(args, model_args, data_args, training_args)
    split = get_split(training_args)
    
    dataset_list = data_args.datasets.split(",")
    episode_indices = get_episode_indices(data_args.episodes)
            
            
    episode_output_dirs = [
        os.path.join(output_dir, f"ep{idx}") for idx in episode_indices if os.path.isdir(os.path.join(output_dir, f"ep{idx}"))
    ]
    
    assert len(episode_indices) == len(episode_output_dirs)

    episode_json_files = []
    episode_csv_files = []

    for ep_dir in episode_output_dirs:
        for dataset in dataset_list:
            episode_json_files += [os.path.join(ep_dir, pos_json) for pos_json in os.listdir(ep_dir) if pos_json.endswith('.json') and f"-{dataset}-" in pos_json]
            episode_csv_files += [os.path.join(ep_dir, pos_csv) for pos_csv in os.listdir(ep_dir) if pos_csv.endswith('.csv') and f"-{dataset}-" in pos_csv]

            logging.warning(f'Reading results from {episode_json_files[0]}')

            results = []
            for i, fn in enumerate(episode_json_files):
                with open(fn) as f:
                    results.append(json.load(f))
                data_df = pd.read_csv(episode_csv_files[i], sep="\t")
                if data_df.shape[0] != data_args.val_examples:
                    logging.warning(f"WARNING: Only found {data_df.shape[0]} examples!")

            # print average results
            res = get_avg_results(results)
            print_results(res)

            # save average results to file
            evaluation_output_filename = get_evaluation_output_filename(data_args)
            dataset_filename = evaluation_output_filename + f'-{dataset}-{split}.json'

            with open(os.path.join(output_dir, dataset_filename), 'w+') as f:
                json.dump(res, f, indent=0)
            
            logging.warning(f'Summary statistics saved in {output_dir}')

            # save key data to csv
            metrics = data_args.dc_metrics.split(",")
            columns = [
                "template_num",
                "mode",
                "model",
                "num_prompts",
                "constrained_decoding",
                "input_format",
                "output_format",
            ] + metrics

            # if making for first time, add column titles
            model_family = get_model_family(model_args.model_name_or_path)
            stat_summary_fn = data_args.dc_filename.replace(data_args.datasets, dataset + f"_{model_family}")
            
            print(stat_summary_fn)
            
            if not os.path.exists(stat_summary_fn):
                with open(stat_summary_fn, "w") as f:
                    f.write("\t".join(columns))    


            line = [
                data_args.dc_template,
                data_args.dc_mode,
                model_args.model_name_or_path,
                str(data_args.num_prompt_ex),
                str(data_args.constrained_decoding),
                str(data_args.input_format),
                str(data_args.output_format),
            ]

            for met in metrics:
                line.append(f"{round(res[met][0], 4)} ({round(res[met][1], 3)})")
                
            with open(stat_summary_fn, 'a') as f:
                f.write("\n" + "\t".join(line))

if __name__ == "__main__":
    main()
