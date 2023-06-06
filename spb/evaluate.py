from transformers import PreTrainedTokenizer
from typing import List, Dict
import logging
import numpy as np
from spb.arguments import DataTrainingArguments
from spb.bench_datasets import load_dataset
from spb.prompting import ICLPromptHelper

def get_avg_results(results: List[dict]) -> dict:
    """
    Compute average results and standard deviation from many episodes.
    """
    aggregate_results = {'num_episodes': len(results)}

    for key in results[0]:
        try:
            numbers = np.array([res[key] for res in results])
            aggregate_results[key] = (numbers.mean(), numbers.std())

        except:
            pass

    return aggregate_results


def print_results(results: dict):
    """
    Print results to stdout.
    """

    for key, value in results.items():
        s = f'{key.replace("_", " "):26} '

        if isinstance(value, (list, tuple)):
            mean, std = value
            s += f'{mean:.6f}\t{std:.6f}'
        elif isinstance(value, float):
            s += f'{value:.6f}'
        else:
            s += f'{value}'

        logging.warning(s)

def evaluate_results(
    results_fn, 
    model, 
    model_name: str, 
    dataset_name: str, 
    data_args: DataTrainingArguments, 
    tokenizer: PreTrainedTokenizer, 
    split: str,
    seed: int, 
    gpu: int, 
    batch_size: int, 
    causal_lm: bool, 
    subset: str
) -> Dict[str, float]:
    """
    Evaluates results from output stored in csv at results_fn.
    """

    test_dataset = load_dataset(
        dataset_name, 
        data_args,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        tokenizer=tokenizer, 
        split=split, 
        seed=seed, 
        shuffle=False, 
        is_eval=True, 
        causal_lm=causal_lm,
        subset=subset
    )

    return test_dataset.evaluate_dataset_results(results_fn, model_name)

def evaluate_icl_causal_lm(
    model, 
    test_dataset, 
    data_args, 
    tokenizer, 
    train_dataset, 
    val_example_list, 
    # device, 
    # head_mask, 
    blurb="", 
    batch_size=1
):
    """
    Perform ICL using a causal LM without using OpenAI API. # TODO: adapt to PromptHelper
    
    """
    # prepare data collection 
    count = 0 
    response_dict = {
                        "example_id": [], 
                        "prompt_len":[], 
                        "prompt": [], 
                        "input": [], 
                        "gt_output": [], 
                        "orig_output": [],
                        "output_sentences": [], 
                        "full_response": [],
                    }
    
    prompt_helper = ICLPromptHelper(
        data_args,
        tokenizer,
        train_dataset,
        blurb=blurb,
        model=model,
    )
    
    
    # Get all prompts
    all_prompts = []

     # remove this
    for j, example_pair in enumerate(val_example_list[:data_args.val_examples]):
        example = example_pair[0]
        prompt = prompt_helper.get_prompt()
        mod_example, prompt_str, responses = prompt_helper.get_causal_lm_response(prompt, example)
        input_str, output_str = prompt_helper.format_example(mod_example)          

        # For each API response
        for full_response in responses:                       
            response = prompt_helper.clean_response(full_response)
            
            # print first 10 examples
            if count < 6:
                count += 1
                print("_______PROMPT_________")
                print(prompt_str)
                print("_______END PROMPT_________")
                print(input_str)
                print(f"Correct Output: {output_str}")
                print(f"Model Output: {response}")

            # save to dictionary
            response_dict["prompt_len"].append(data_args.num_prompt_ex)
            response_dict["prompt"].append(prompt_str)
            response_dict["input"].append(input_str)
            response_dict["example_id"].append(mod_example.id)
            
            if train_dataset.is_classification_dataset:
                response_dict["gt_output"].append(mod_example.gt_label_name)
                
                # if classification 
                if data_args.replace_labels is not None:
                    response_dict["orig_output"].append(mod_example.orig_gt_label_name)
                    
                else:
                    response_dict["orig_output"].append("N/A")
            else:
                response_dict["gt_output"].append(output_str)
                response_dict["orig_output"].append("N/A")
                
            response_dict["output_sentences"].append(response)
            response_dict["full_response"].append(full_response)

    return response_dict

def evaluate_icl_gpt3(
    model, 
    test_dataset, 
    data_args, 
    tokenizer, 
    train_dataset, 
    val_example_list, 
    blurb=""
):
    """
    Perform ICL using OpenAI API.
    """
    
    # prepare data collection 
    count = 0 
    response_dict = {
                        "example_id": [], 
                        "prompt_len":[], 
                        "prompt": [], 
                        "input": [], 
                        "gt_output": [], 
                        "orig_output": [],
                        "output_sentences": [], 
                        "full_response": [],
                    }

    gpt3_helper = ICLPromptHelper(
        data_args,
        tokenizer,
        train_dataset,
        blurb=blurb,
        gpt3_model_name=model,
    )
    
    # remove this
    for j, example_pair in enumerate(val_example_list[:data_args.val_examples]):
        example = example_pair[0]
        prompt = gpt3_helper.get_prompt()
        mod_example, prompt_str, responses = gpt3_helper.get_openai_response(prompt, example)
        input_str, output_str = gpt3_helper.format_example(mod_example)          
            
        # For each API response
        for full_response in responses:                       
            response = gpt3_helper.clean_response(full_response)
            
            # print first 10 examples
            if count < 10:
                count += 1
                print("_______PROMPT_________")
                print(prompt_str)
                print("_______END PROMPT_________")
                print(input_str)
                print(f"Correct Output: {output_str}")
                print(f"Model Output: {response}")

            # save to dictionary
            response_dict["prompt_len"].append(data_args.num_prompt_ex)
            response_dict["prompt"].append(prompt_str)
            response_dict["input"].append(input_str)
            response_dict["example_id"].append(mod_example.id)
            
            if train_dataset.is_classification_dataset:
                response_dict["gt_output"].append(mod_example.gt_label_name)
                
                # if classification 
                if data_args.replace_labels is not None:
                    response_dict["orig_output"].append(mod_example.orig_gt_label_name)
                    
                else:
                    response_dict["orig_output"].append("N/A")
            else:
                response_dict["gt_output"].append(output_str)
                response_dict["orig_output"].append("N/A")
                
            response_dict["output_sentences"].append(response)
            response_dict["full_response"].append(full_response)

    return response_dict

def evaluate_icl(model, train_dataset, dataset_name: str, data_args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, split: str,
             seed: int, gpu: int, causal_lm: bool, head_mask=None, subset=None):
    """
    Handles which ICL function to use for this particular case.
    """
    test_dataset = load_dataset(
        dataset_name, 
        data_args,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        tokenizer=tokenizer, 
        split=split, 
        seed=seed, 
        shuffle=True, 
        is_eval=True, 
        causal_lm=causal_lm,
        subset=subset
    )

    if data_args.val_examples == -1 or data_args.val_examples > len(test_dataset):
        data_args.val_examples = len(test_dataset)

    val_example_list = test_dataset.get_icl_examples(
        data_args.val_examples,
        max_len_per_ex = int(data_args.max_seq_length / data_args.num_prompt_ex),
    ) 

    if causal_lm:
        model.eval()
        return evaluate_icl_causal_lm(
            model, 
            test_dataset, 
            data_args, 
            tokenizer, 
            train_dataset, 
            val_example_list, 
        )

    else:
        return evaluate_icl_gpt3(
            model, 
            test_dataset, 
            data_args, 
            tokenizer, 
            train_dataset, 
            val_example_list
        )
