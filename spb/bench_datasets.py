import string
import logging
import os
import random
import logging
import numpy as np
import pandas as pd
import datasets as ds

from collections import Counter
from tqdm import tqdm
from transformers import PreTrainedTokenizer 
from typing import Dict, List

from spb.arguments import DataTrainingArguments
from spb.base_dataset import BaseDataset
from spb.input_example import InputExample
from spb.utils import add_period_for_lmbff, get_edit_distance 



DATASETS = {}

def register_dataset(dataset_class):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
        dataset_name: str,
        data_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        split: str,
        max_input_length: int,
        max_output_length: int,
        num_examples: int = -1,
        seed: int = None,
        shuffle: bool = True,
        is_eval: bool = False,
        causal_lm: bool = False,
        subset: str = None,
        do_train: bool = False,
    ):
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        mode=split,
        overwrite_cache=data_args.overwrite_cache,
        num_examples=num_examples,
        seed=seed,
        shuffle=shuffle,
        data_args=data_args,
        is_eval=is_eval,
        causal_lm=causal_lm,
        subset=subset,
        do_train=do_train,
    )

class ClassificationDataset(BaseDataset):
    """
    Base class for text classification datasets.
    """
    default_output_format = "classification"
    is_classification_dataset=True
    to_tf = {
        "entailment": "True",
        "contradiction": "False",
        "neutral": "Unknown",
        "Unknown": "Unknown",
        "Yes": "True",
        "No": "False",
    }
    
    def get_example_indices(self, n: int, legal_labels: List[str] = None, cover_label_space=True) -> List[int]:

        """
        For train, return list of examples that are guaranteed to have legally labelled entities (i.e. no mislabelled examples 
        and all labels exist in the legal label list).
        
        mandated_labels: guarantee that at least one example in each of the labels is shown.

        For eval, just return list of examples.
        """

        shuffled_indices = self.indices.copy()
        
        if self.mode != "train" and self.seed is not None:
            random.seed(self.seed)
            random.shuffle(shuffled_indices)
            random.seed()
            
        else:
            random.shuffle(shuffled_indices)

        if self.mode == "train":

            if cover_label_space:
                labels_counter = {
                    k: 0 for k in self.natural_entity_types.keys()
                }

                labels_idx= {
                    k: [] for k in self.natural_entity_types.keys()
                }

            idx = []
            count = 0

            while len(idx) < n:
                shuffled_idx = shuffled_indices[count]
                example = self.get_example(shuffled_idx)

                # make sure entities isn't empty list and all labels are legal
                if self.ensure_legal_example(example, legal_labels=legal_labels):
                    idx.append(shuffled_idx)

                    # count this label in the label counter, if necessary
                    if cover_label_space:
                        labels_counter[example.gt_label] += 1
                        labels_idx[example.gt_label].append(shuffled_idx)

                count += 1

            try:
                # make sure entire label space is covered
                if cover_label_space: # and 0 in labels_counter.values():
                    balanced_label_number = round(n/len(self.natural_entity_types.keys()))
                    # if not, remove some examples and replace them with the missed labels
                    missed_labels = [k for k in labels_counter.keys() if labels_counter[k] < balanced_label_number]
                    
                    # while we're still missing labels
                    while len(missed_labels) > 0:
                        shuffled_idx = shuffled_indices[count]
                        example = self.get_example(shuffled_idx)
                        extra_example_labels = sorted(
                            [k for k in labels_counter.keys() if labels_counter[k] >= balanced_label_number], 
                            key=lambda k: labels_counter[k], 
                        )

                        # check to see if example is legal + is a missed label
                        if self.ensure_legal_example(example, legal_labels=legal_labels) and example.gt_label in missed_labels:
                            idx.append(shuffled_idx)
                            labels_counter[example.gt_label] += 1
                            labels_idx[example.gt_label].append(shuffled_idx)

                            if labels_counter[example.gt_label] >= balanced_label_number:
                                missed_labels.remove(example.gt_label)

                            # remove an extra example from a different label class and adjust the idx/counter accordingly
                            extra_example = labels_idx[extra_example_labels[-1]][-1]
                            idx.remove(extra_example)
                            labels_idx[extra_example_labels[-1]].remove(extra_example)
                            labels_counter[extra_example_labels[-1]] -= 1
                        
                        count +=1

                    random.shuffle(idx)   
            except:               
                random.shuffle(idx) 
        else:
            idx = shuffled_indices[:n]

        return idx

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None, gt_output=None, orig_output=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract entities and relations from output sentence
        predicted_class = self.output_format.run_inference(
            example,
            output_sentence
        )
        
        
        # correct_prediction = int(self.output_format.format_output(example) == predicted_class)
        correct_prediction = int(gt_output == predicted_class)

                 
        if self.data_args.replace_labels is not None:
            original_correct_prediction = int(predicted_class == orig_output)
        
        else:
            original_correct_prediction = correct_prediction

        res = Counter({
            'correct_prediction': correct_prediction,
            'original_correct_prediction': original_correct_prediction,
            'valid_output': int(predicted_class in self.natural_entity_types.values()),
            'num_sentences': 1
        })

        return res

    
    def evaluate_dataset_results(self, results_fn: str, model_name: str, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, given a filename to a \t-separated csv with results.
        """
        print("Reading results from {}".format(results_fn))
        
        #initialize results counter variable and read in output file
        results = Counter()
        # edit_distances = []
        output_sentences_df = pd.read_csv(results_fn, sep="\t")

        for index, row in output_sentences_df.iterrows():
            example = self.get_example(int(0))

            if str(row["output_sentences"]).strip() == "" or type(row["output_sentences"]) != str:
                output_sentence = str(row["full_response"]).strip().split("\n")[0]
            else:
                output_sentence = str(row["output_sentences"]).strip()

            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence.strip(),
                    gt_output=str(row["gt_output"]),
                    orig_output=str(row["orig_output"]) if "orig_output" in row.keys() else "N/A",
                    model=None,
                    tokenizer=self.tokenizer,
                )
            results += new_result

            # calculate edit_distance
            # edit_distance = get_edit_distance(self.output_format.format_output(example), str(row["output_sentences"]))
            # edit_distances.append(edit_distance)

        accuracy = results["correct_prediction"]/results["num_sentences"]
        original_accuracy = results["original_correct_prediction"]/results["num_sentences"]
        valid_output = results["valid_output"]/results["num_sentences"]

        res = {
            'accuracy': accuracy,
            'original_accuracy': original_accuracy,
            'valid_output': valid_output,
            # 'normalized_edit_distance': np.mean(edit_distances),
        }

        return res


    def randomize_example(self, example, span_labels):
        
        # randomize span label
        if self.data_args.random_label:
            example.gt_label_name = random.choice(span_labels)
            
        return example
    
    
    def change_label_space(self, entity_type_dict=None):
        """
        Returns a dictionary which maps the original label space to a different label space
        """
        n_labels = len(list(self.natural_entity_types.values()))
        new_entity_types = self.natural_entity_types.copy()
        
        if self.data_args.label_space == "abstract":
            span_labels = list("@#$%*^\{\}") 

        elif self.data_args.label_space == "dummy":
            span_labels = ["dummy" for i in range(n_labels)]
        
        elif self.data_args.label_space == "number":
            span_labels = [str(i) for i in range(n_labels)]  
        
        elif self.data_args.label_space == "letter":
            span_labels = random.sample(string.ascii_uppercase, n_labels)

        random.shuffle(span_labels)
        
        for i, k in enumerate(self.natural_entity_types.keys()):
            new_entity_types[k] = span_labels[i] 
            
        return new_entity_types
                
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.csv')

        if not os.path.exists(file_path):
            logging.warning("Downloading data...")
            self.save_hf_data(split, file_path)
            
        assert os.path.exists(file_path)
        data = pd.read_csv(file_path).sample(frac=1)
        
        # load in examples:
        examples = []
        for i in tqdm(range(min(self.data_args.max_dataset_size, data.shape[0]))):
            sentence =  data["text"][i]
            label = data["label"][i]
            gt_label_name=self.natural_entity_types[label]

            try:
                example = InputExample(
                        id=f'{split}-{i}',
                        dataset=self.name,
                        tokens=sentence.split(" "),
                        gt_label=label,
                        gt_label_name=gt_label_name,
                    )
                
                examples.append(example)
            except:
                pass
        
        logging.warning(f"Loaded {len(data)} sentences for split {split} of {self.name}")
        return examples

    def save_hf_data(self, split, file_path):
        sentences, labels = self._save_hf_data(split)
        
        df = pd.DataFrame({
            "text": sentences, 
            "label": labels,
        })
        
        df.to_csv(file_path)
        
        
          
@register_dataset
class EmotionDataset(ClassificationDataset):
    name = "emotion"
    task = "sentiment"
    
    def load_schema(self): 

        self.natural_entity_types = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        }
        self.unnatural_entity_types = {
            0: "sad",
            1: "joy",
            2: "love",
            3: "ang",
            4: "fear",
            5: "surp",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        data = ds.load_dataset('emotion')
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        return data["text"], data["label"]
    

class NLIDataset(ClassificationDataset):
    task = None
    
    def save_hf_data(self, split, file_path):
        sent1, sent2, labels = self._save_hf_data(split)
        
        df = pd.DataFrame({
            "sent1": sent1, 
            "sent2": sent2, 
            "label": labels,
        })
        
        df.to_csv(file_path)
    
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.csv')

        if not os.path.exists(file_path) or self.data_args.overwrite_hf_data:
            logging.warning("Downloading data...")
            self.save_hf_data(split, file_path)
            
        assert os.path.exists(file_path)
        data = pd.read_csv(file_path).sample(frac=1)
        
        # load in examples:
        examples = []
        for i in tqdm(range(min(self.data_args.max_dataset_size, data.shape[0]))):
            sent1 =  data["sent1"][i]
            sent2 =  data["sent2"][i]
            label = data["label"][i]
            gt_label_name=self.natural_entity_types[label]

            try:
                example = InputExample(
                        id=f'{split}-{i}',
                        dataset=self.name,
                        sentence1=sent1,
                        sentence2=sent2,
                        gt_label=label,
                        gt_label_name=gt_label_name,
                    )
                
                examples.append(example)
            except:
                pass
        
        logging.warning(f"Loaded {len(data)} sentences for split {split} of {self.name}")
        return examples
    
@register_dataset
class SNLIDataset(NLIDataset):
    name = "snli"
    default_input_format = "nli1"
    task = "nli"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "Yes",
            1: "Unknown",
            2: "No"
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        data = ds.load_dataset('snli')
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        data = data.filter(lambda example: example["label"] != -1)
          
        return data["premise"], data["hypothesis"], data["label"]
          

@register_dataset
class SST2Dataset(ClassificationDataset):
    name = "sst2"
    default_input_format = "sentiment1"
    task = "sentiment"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "negative",
            1: "positive",
        }
        
        self.unnatural_entity_types = {
            0: "neg",
            1: "pos",
        }

        self.lmbff_entity_types = {
            0: "terrible",
            1: "great",
        }

    def _save_hf_data(self, split):
        data = ds.load_dataset('sst2')
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        return data["sentence"], data["label"]
    
@register_dataset
class TRECDataset(ClassificationDataset):
    name = "trec"
    default_input_format = "topic1"
    task = "topic"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "abbreviation",
            1: "entity",
            2: "description",
            3: "human",
            4: "location",
            5: "number",
        }
        
        self.unnatural_entity_types = {
            0: "abb",
            1: "ent",
            2: "desc",
            3: "per",
            4: "loc",
            5: "num",
        }

        self.lmbff_entity_types = {
            0: "expression",
            1: "entity",
            2: "description",
            3: "human",
            4: "location",
            5: "number",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        if split == "test":
            data = ds.load_dataset('trec', split="test")

        elif split == "dev":
            data = ds.load_dataset('trec', split="train[80%:]")

        else:
            data = ds.load_dataset('trec', split="train[:80%]")
        
        return data["text"], data["coarse_label"]
   
class GlueNLIDataset(NLIDataset):
     
    def get_glue_data(self, split: str, category: str):
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        
        if split == "dev":
            data = ds.load_dataset('glue', category, split="validation")
        else:
            data = ds.load_dataset('glue', category, split="train")

        sent1=[]
        sent2=[]
        labels=[]
        
        for i in tqdm(range(min(self.data_args.max_dataset_size, len(data["label"])))):
            sent1.append(data["sentence1"][i])
            sent2.append(data["sentence2"][i])
            labels.append(data["label"][i])
            
        return sent1, sent2, labels
    

@register_dataset
class RTEDataset(GlueNLIDataset):
    name = "rte"
    default_input_format = "nli_entailment1"
    task = "nli_entailment"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "Yes",
            1: "No",
        }
 
    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        return self.get_glue_data(split, "rte")
    

@register_dataset
class WNLIDataset(GlueNLIDataset):
    name = "wnli"
    default_input_format = "nli_entailment1"
    task = "nli_entailment"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "No",
            1: "Yes",
        }
        
    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        return self.get_glue_data(split, "wnli")
    
@register_dataset
class MRPCDataset(GlueNLIDataset):
    name = "mrpc"
    default_input_format = "paraphrase1"
    task = "paraphrase"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "No",
            1: "Yes",
        }

    def _save_hf_data(self, split: str):
        return self.get_glue_data(split, "mrpc")
    
@register_dataset
class PoemSentimentDataset(ClassificationDataset):
    name = "poem"
    default_input_format = "sentiment1"
    task = "sentiment"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "negative",
            1: "positive",
            2: "no impact",
            3: "mixed",
        }
        
        self.unnatural_entity_types = {
            0: "neg",
            1: "pos",
            2: "none",
            3: "mix",
        }
        
        self.lmbff_entity_types = {
            0: "terrible",
            1: "great",
            2: "neutral",
            3: "mixed",
        }
        

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        data = ds.load_dataset('poem_sentiment')
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        return data["verse_text"], data["label"]

    
@register_dataset
class TweetEvalEmotionDataset(ClassificationDataset):
    name = "tweet_eval_emotion"
    default_input_format = "sentiment1"
    task = "sentiment"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "anger",
            1: "joy",
            2: "optimism",
            3: "sadness",
        }
        
        self.unnatural_entity_types = {
            0: "ang",
            1: "joy",
            2: "opt",
            3: "sad",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        data = ds.load_dataset('tweet_eval', "emotion")
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]


        return data["text"], data["label"]

@register_dataset
class TweetEvalHateDataset(ClassificationDataset):
    name = "tweet_eval_hate"
    default_input_format = "hate_speech1"
    task = "hate_speech"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "No",
            1: "Yes",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        data = ds.load_dataset('tweet_eval', "hate")
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        return data["text"], data["label"]
    

@register_dataset
class TweetEvalAtheismDataset(ClassificationDataset):
    name = "tweet_eval_atheism"
    default_input_format = "tweet_atheism1"
    task = "tweet_atheism"
    
    def load_schema(self): 
        self.official_entity_types = {
            0: "none",
            1: "against",
            2: "favor", # with respect to atheism
        }
        
        self.natural_entity_types = {
            0: "Unknown",
            1: "False",
            2: "True",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        data = ds.load_dataset('tweet_eval', "stance_atheism")
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        return data["text"], data["label"]
    

@register_dataset
class TweetEvalFeministDataset(ClassificationDataset):
    name = "tweet_eval_feminist"
    default_input_format = "tweet_feminist1"
    task = "tweet_feminist"
    
    def load_schema(self): 
        self.official_entity_types = {
            0: "none",
            1: "against",
            2: "favor",
        }
        
        self.natural_entity_types = {
            0: "Unknown",
            1: "False",
            2: "True",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        data = ds.load_dataset('tweet_eval', "stance_feminist")
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        return data["text"], data["label"]
    
@register_dataset
class AmazonPolarityDataset(ClassificationDataset):
    name = "amazon_polarity"
    task = "sentiment"
    default_input_format = "sentiment1"
    
    def load_schema(self): 
        self.official_entity_types = {
            0: "none",
            1: "against",
        }
        
        self.natural_entity_types = {
            0: "negative",
            1: "positive",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        
        if split == "test":
            data = ds.load_dataset('amazon_polarity', split="test").shuffle()

        elif split == "validation":
            data = ds.load_dataset('amazon_polarity', split="train[:80%]").shuffle()

        else:
            data = ds.load_dataset('amazon_polarity', split="train[80%:]").shuffle()

        return data["text"], data["label"]
    
@register_dataset
class HateSpeech18Dataset(ClassificationDataset):
    name = "hate_speech18"
    default_input_format = "hate_speech1"
    task = "hate_speech"
    def load_schema(self): 

        self.official_entity_types = {
            0: "hate",
            1: "noHate",
        }


        self.natural_entity_types = {
            0: "Yes",
            1: "No",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        annotations = pd.read_csv(f"data/{self.name}/annotations_metadata.csv")
        annotations = annotations.sample(frac=1)

        text = []
        labels = []

        label_to_id = {
            v: k for k, v in self.official_entity_types.items()
        }

        for index, row in tqdm(annotations.iterrows()):
            file_id = row["file_id"]
            fn = f"data/{self.name}/all_files/{file_id}.txt"
            label = row["label"] 

            if label in label_to_id.keys():
                labels.append(label_to_id[label])
                
                with open(fn, "r") as f:
                    text.append(f.read())

        assert len(text) == len(labels)

        return text, labels

@register_dataset
class SICKDataset(NLIDataset):
    name = "sick"
    default_input_format = "nli1"
    task = "nli"
    
    def load_schema(self): 
        
        self.natural_entity_types = {
            0: "Yes",
            1: "Unknown",
            2: "No",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        data = ds.load_dataset("sick")
        
        if split == "test":
            data = data["test"]

        elif split == "dev":
            data = data["validation"]

        else:
            data = data["train"]

        sent1 = []
        sent2 = []
        labels = []

        for i in tqdm(range(min(self.data_args.max_dataset_size, len(data["label"])))):
            sent1.append(add_period_for_lmbff(data["sentence_A"][i]))
            sent2.append(add_period_for_lmbff(data["sentence_B"][i]))
            labels.append(data["label"][i])

        return sent1, sent2, labels

@register_dataset
class FinancialPhraseBank(ClassificationDataset):
    name = "financial_phrasebank"
    default_input_format = "sentiment1"
    task = "sentiment"
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """

        data = ds.load_dataset("financial_phrasebank", "sentences_allagree")["train"].shuffle()
        
        if split == "test":
            data = data[:round(data.num_rows*0.25)]

        elif split == "dev":
            data = data[round(data.num_rows*0.25):round(data.num_rows*0.5)]

        else:
            data = data[round(data.num_rows*0.5):]

        return data["sentence"], data["label"]
        

class EthosDataset(ClassificationDataset):
    
    def load_schema(self): 
        self.natural_entity_types = {
            0: "No",
            1: "Yes",
        }

    def get_ethos_data(self, split: str, category: str):
        """
        Load data for a single split (train, dev, or test).
        """

        data = ds.load_dataset("ethos", "multilabel")["train"].shuffle()
        
        if split == "test":
            data = data[:round(data.num_rows*0.25)]

        elif split == "dev":
            data = data[round(data.num_rows*0.25):round(data.num_rows*0.5)]

        else:
            data = data[round(data.num_rows*0.5):]

        return data["text"], data[category]
    
@register_dataset    
class EthosRaceDataset(EthosDataset):
    name = "ethos_race"
    default_input_format = "hate_speech_race1"
    task = "hate_speech_race"
    
    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        return self.get_ethos_data(split, "race")
    
@register_dataset    
class EthosGenderDataset(EthosDataset):
    name = "ethos_gender"
    default_input_format = "hate_speech_gender1"
    task = "hate_speech_gender"
    
    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        return self.get_ethos_data(split, "gender")

@register_dataset    
class EthosReligionDataset(EthosDataset):
    name = "ethos_religion"
    default_input_format = "hate_speech_religion1"
    task = "hate_speech_religion"
    
    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        return self.get_ethos_data(split, "religion")
    
@register_dataset    
class EthosNationalOriginDataset(EthosDataset):
    name = "ethos_national_origin"
    default_input_format = "hate_speech_national1"
    task = "hate_speech_national"
    
    def _save_hf_data(self, split: str):
        """
        Load data for a single split (train, dev, or test).
        """
        return self.get_ethos_data(split, "national_origin")