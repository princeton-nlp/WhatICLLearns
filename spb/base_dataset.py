import os
import copy
import logging
import random
from typing import Tuple, List
from abc import ABC, abstractmethod
import torch
from torch.utils.data.dataset import Dataset
import string
from transformers import PreTrainedTokenizer, torch_distributed_zero_first

from spb.arguments import DataTrainingArguments
from spb.input_example import InputFeatures, InputExample
from spb.input_formats import INPUT_FORMATS
from spb.output_formats import OUTPUT_FORMATS

class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset
    data_name = None    # name of the directory, if different from the name of the dataset
    task_descriptor = None  # string to prepend to every input sentence if multitask=True (default is self.name)
    task = None
    entity_types = None
    is_classification_dataset = False
    natural_entity_types = None
    to_tf = None

    default_input_format = 'plain'
    default_output_format = None
    default_data_dir = 'data'

    def __init__(
            self,
            data_args: DataTrainingArguments,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            overwrite_cache: bool = False,
            mode: str = 'train',
            local_rank: int = -1,
            num_examples: int = -1,
            seed: int = None,
            shuffle: bool = True,
            is_eval: bool = False,
            causal_lm=True,
            subset: str = None,
            do_train: bool = False,
    ):
        # if seed is not None:
        #     # set random seed for repeatability
        #     random.seed(seed)

        # set various class attributes
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.causal_lm = causal_lm # is 
        self.num_examples = num_examples
        self.subset = subset
        self.do_train = do_train
        self.mode = mode
        self.seed = seed

        self.input_format = INPUT_FORMATS[
            data_args.input_format if data_args.input_format is not None else self.default_input_format
        ]()

        self.output_format = OUTPUT_FORMATS[
            data_args.output_format if data_args.output_format is not None else self.default_output_format
        ]()

        self.output_format.seed = self.seed

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir

        self.is_eval = is_eval
        # self.eval_nll = data_args.eval_nll
        
        # If template is supplied, override default input/output formats
        if self.data_args.natlan_template_number is not None:
            # handle natural language labels:
            template_input_format = self.task + self.data_args.natlan_template_number
            self.input_format = INPUT_FORMATS[template_input_format]()
        
        if self.data_args.minimal_template_number is not None:
            # multi-sentence datasets
            if "nli" in self.task or "paraphrase" in self.task:
                template_input_format = "nli_minimal" + self.data_args.minimal_template_number
            else:
                template_input_format = "minimal" + self.data_args.minimal_template_number
            self.input_format = INPUT_FORMATS[template_input_format]()
    

        # set delimiter for datasets that need truncating
        if causal_lm:
            self.delimiter = "Ġ"
        else:
            self.delimiter = "_"

        cached_data_file = os.path.join(
            self.data_dir(),
            f"cached_{self.name}_{mode}_{tokenizer.__class__.__name__}_{max_input_length}_{max_output_length}.pth"
        )

        # self.natural_entity_types = None                            
        
        try:
            os.mkdir(self.data_dir())
        except FileExistsError:
            pass

        with torch_distributed_zero_first(local_rank):
            # make sure only the first process in distributed training processes the dataset,
            # and the others can use the cached version

            if os.path.exists(cached_data_file) and not overwrite_cache:
                logging.warning(f"Loading cached data...")
                self.load_cached_data(cached_data_file)

            else:
                logging.warning(f"Loading data from scratch...")

                self.load_schema()   # here the dataset can load information such as entity/relation types
                
                # if not self.is_classification_dataset and data_args.replace_labels is not None:
                #     self.map_example_labels()
                
                # map label words to True/False if necessary
                if self.input_format.convert_to_true_false:
                    tf_entity_types = {}
                    for k, v in self.natural_entity_types.items():
                        tf_entity_types[k] = self.to_tf[v]
                        
                    self.natural_entity_types = tf_entity_types

            
                self.examples = self.load_data(mode=mode, seed=seed)

                logging.warning("Assigning examples to this dataset...")
                
                # assign examples to this dataset
                for example in self.examples:
                    example.dataset = self

                # reduce token sequence length
                if data_args.input_seq_length != -1:
                    for example in self.examples:
                        if len(example.tokens) >  data_args.input_seq_length:
                            example.tokens = example.tokens[:data_args.input_seq_length]

                # # For Shorter Entity Span Experiments: replace entity spans with simpler name
                # if data_args.abstract_names and self.mode == "train":
                #     self.replace_example_spans()


                self.features = [0]

                if local_rank in [-1, 0]:
                    # save data
                    logging.warning("Saving data...")
                    try:
                        self.save_data(cached_data_file)
                    except Exception as e:
                        print(e)

            # set default indices
            self.indices = list(range(len(self.examples)))
            
            # otherwise, if necessary, shuffle the indices
            if seed is not None and shuffle and self.mode != "train":
                random.seed(self.seed)
                random.shuffle(self.indices)
                random.seed()

            # Make sure that data indices are randomly shuffled.
            # Then, fix the seed using self.seed.
            elif seed is not None and shuffle and self.mode == "train":
                random.shuffle(self.indices)

            # if necessary reduce effective dataset size
            if self.num_examples == -1:
                self.effective_size = len(self.examples)
                logging.warning(f"Effective dataset size is {self.effective_size}")

            else:
                self.effective_size = min(self.num_examples, len(self.examples))
                logging.warning(f"Effective dataset size reduced to {self.effective_size}")

    
    def __repr__(self):
        return f'Dataset {self.name}'

    def __len__(self):
        return self.effective_size

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[self.indices[i]]

    def get_example(self, i: int) -> InputExample:
        return self.examples[self.indices[i]]

    def ensure_legal_example(self, example, legal_labels=None):
        if example.entities is None or len(example.entities) > 0 and (legal_labels == None or all([ent.type.natural in legal_labels for ent in example.entities])):
            return True
        else:
            return False

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
            idx = []
            count = 0

            while len(idx) < n:
                shuffled_idx = shuffled_indices[count]
                example = self.get_example(shuffled_idx)

                # make sure entities isn't empty list and all labels are legal
                if self.ensure_legal_example(example, legal_labels=legal_labels):
                    idx.append(shuffled_idx)

                count += 1

        else:
            idx = shuffled_indices[:n]

        return idx
        
    # def replace_example_spans(self):
    #     """
    #     Replace the original example entity names with simpler + shorter ones.
    #     """
    #     span_names = {
    #                 "location": ["China", "Mexico", "Brazil", "London", "Paris", "Boston", "Milan", "Italy", "England", "Russia", "Canada", "Japan", "Spain", "Scotland", "Ireland", "Britain", "India", "Utah", "California", "Florida"],
    #                 "person": ["John", "Jane", "Howard", "Steve", "Tom", "Frank", "Emily", "Patrick", "Sarah", "Bob", "Tim", "Jim", "Kate", "Pat", "Fred", "Kim", "George", "Alex", "Jen", "Ann", "Will", "Austin", "Steve", "Jess"],
    #                 "organization": ["BBC", "Amazon", "Google", "Microsoft", "Congress", "Netflix", "Apple", "Tesla", "Facebook", "Samsung", "Senate", "NPR", "WHO", "NBC", "Shell", "JD", "Parliament", "Alibaba", "Tencent", "Walmart"],
    #                 "miscellaneous": ["1990", "March", "2022", "Febuary", "1992", "2010", "1879", "1999", "January", "December", "April", "June", "July", "1998", "August", "September", "November", "2004", "1940", "1932", "1929"],
    #                 "other": ["1990", "March", "2022", "Febuary", "1992", "2010", "1879", "1999", "January", "December", "April", "June", "July", "1998", "August", "September", "November", "2004", "1940", "1932", "1929"],
    #                 }

    #     for example in self.examples:
    #         span_idx = {k: 0 for k in span_names.keys()}
    #         span_lengths = [ent.end-ent.start-1 for ent in example.entities]
    #         for i, ent in enumerate(example.entities):
    #             ent.start -= sum(span_lengths[:i])
    #             ent.end = ent.start + 1
    #             example.tokens[ent.start] = span_names[ent.type.natural][span_idx[ent.type.natural]]
    #             span_idx[ent.type.natural] +=1 
                
    #             if span_lengths[i] == 1:
    #                 del example.tokens[ent.start+1]
    #             elif span_lengths[i] > 1:
    #                 del example.tokens[ent.start+1:ent.start+span_lengths[i]+1]


    def map_example_labels(self):
        """
        Replace the official natural language labels some kind of other label.
        """
        self.original_entity_types = self.entity_types.copy()
        n_labels = range(len(list(self.natural_entity_types.values())))
        if self.data_args.replace_labels == "abstract":
            span_labels = "@#$%{}"
        
        # elif self.data_args.replace_labels == "permute": # TODO: right now I fix the seed so the permutation is the same across train/dev. but prob should be randomized per prompt.
        #     span_labels = list(self.natural_entity_types.values()).copy()
        #     n = self.seed+1
        #     while span_labels == list(self.natural_entity_types.values()):
        #         span_labels = span_labels[-1*n:] + span_labels[:-1*n]
        #         n +=1 
        #     logging.warning(f"The new span labels are: {span_labels}")
            
        elif self.data_args.replace_labels == "random_map":
            span_labels = list(self.natural_entity_types.values()).copy()
            
            while span_labels == list(self.natural_entity_types.values()):
                random.shuffle(span_labels)
            
        elif self.data_args.replace_labels == "number":
            span_labels = [str(i) for i in n_labels]       
            random.seed(self.seed)     
            random.shuffle(span_labels)
            random.seed()

        elif self.data_args.replace_labels == "letter":
            span_labels = random.sample(string.ascii_uppercase, n_labels)
            
        elif self.data_args.replace_labels == "dummy":
            span_labels = ["dummy" for i in n_labels]        

        for i, k in enumerate(self.entity_types.keys()):
            self.entity_types[k].natural = span_labels[i]
        
        
        logging.warning(f"The new span labels are: {self.entity_types}")
            
        
    def get_random_indices(self, example, num_ents):
        random_indices = sorted(random.sample(range(len(example.tokens)), num_ents))
        indices = []
        
        # get random span lengths
        for idx in random_indices:
            indices.append(idx)
            l = random.choice(range(1, 5))
            
            if idx + l > len(example.tokens): 
                indices.append(len(example.tokens))
            else:
                indices.append(idx + l)

        # remove duplicates
        indices = sorted(list(set(indices)))
        
        # if number of indices is odd, remove the last index
        if len(indices) % 2 == 1 and len(indices) > 2:
            indices = indices[:-1]
        
        num_random_ents = int(len(indices)/2)
        return [(indices[2*x], indices[2*x+1]) for x in range(num_random_ents)], num_random_ents

    # def get_perturbed_indices(self, example, num_ents):
    #     random_indices = [(ent.start, ent.end) for ent in example.entities]

    #     indices = []
    #     # get random span lengths
    #     for idx in random_indices:
    #         p = random.choice(range(-1, 2))
    #         pos = random.choice(range(2))
            
    #         indices.append(idx[pos]+p)
    #         indices.append(idx[1-pos])

    #     # remove duplicates and illegal indices
    #     indices.sort()
    #     for i, idx in enumerate(indices):

    #         # if first index is less than 0
    #         if i == 0 and indices[i] < 0:
    #             indices[i] = 0
    #             indices = remove_endpoint_duplicates(indices, first=True)

    #         # if last index goes beyond example length
    #         elif i == len(indices)-1 and indices[i] > len(example.tokens):
    #             indices[i] = len(example.tokens)

    #         else:
    #             # check to see if index collides with previous index, and shift it if so
    #             if i > 0 and indices[i] == indices[i-1]:
    #                 indices[i] = indices[i]+1
    #                 indices = remove_endpoint_duplicates(indices, first=False)

    #     # ideally there should still be the same number of spans
    #     num_random_ents = int(len(indices)/2)
    #     assert(num_random_ents == num_ents)
        
    #     return [(indices[2*x], indices[2*x+1]) for x in range(num_random_ents)]
    
    # def get_label_map(self, span_labels):
    #     random.seed(self.seed)
    #     label_map = dict(zip(span_labels, random.shuffle(span_labels)))
    #     reverse_label_map = {v: k for k, v in label_map}
    #     return label_map, reverse_label_map

    # def get_mixed_random_indices(self, example, num_ents):
        
    #     num_random_ents = num_ents // 2
        
    #     if num_random_ents == 0:
    #         random_spans = []
    #         random_ent_indices = []

    #     else:
    #         random_ent_indices = random.sample(range(len(example.entities)), num_random_ents)
    #         true_ent_spans = [(ent.start, ent.end) for i, ent in enumerate(example.entities) if i not in random_ent_indices]

    #         print("example: {}".format(" ".join(example.tokens)))
    #         print("output: {}".format(self.output_format.format_output(example)))
            
    #         try:
    #             # get list of spans that are not currently occupied by true ents
    #             legal_random_spans = []
    #             for i, (ent_start, ent_end) in enumerate(true_ent_spans):
    #                 if i == 0:
    #                     span_start = 0
    #                     span_end = ent_start

    #                 elif i == len(true_ent_spans):
    #                     span_start = ent_end
    #                     span_end = len(example.tokens)

    #                 else:
    #                     span_start = next_span_start
    #                     span_end = ent_start
                    
    #                 next_span_start = ent_end
    #                 if span_end - span_start > 0:
    #                     legal_random_spans.append((span_start, span_end))

                
    #             # get random spans            
    #             random_spans = []
    #             for j in range(num_random_ents):
                    
    #                 # choose which span to add 
    #                 legal_span_start, legal_span_end = random.choice(legal_random_spans)
    #                 legal_span_end += 1
                    
    #                 if legal_span_end-legal_span_start <2:
    #                     import pdb; pdb.set_trace()
    #                 span_start, span_end = sorted(random.sample(range(legal_span_start, legal_span_end), 2))
    #                 random_spans.append((span_end, legal_span_end))
    #         except Exception as e:
    #             print(e)                
    #             import pdb; pdb.set_trace()

    #     return random_spans, random_ent_indices
        
                    
    def randomize_example(self, example, span_labels):
        random_example = copy.deepcopy(example)

        num_ents = len(example.entities)
        random_ents = range(len(example.entities)) # list of which entities to randomize
                                                   # defaults to ALL

        # # generate random spans
        # if self.data_args.random_span:
        #     num_ents = max(len(example.entities), 1)
        #     random_index_pairs, num_random_ents = self.get_random_indices(example, num_ents)
        #     random_ents = range(num_random_ents)
        
        # # perturb spans
        # elif self.data_args.perturb_span:
        #     random_index_pairs = self.get_perturbed_indices(example, num_ents)
        
        # elif self.data_args.mixed_random_span:
        #     random_index_pairs, random_ents = self.get_mixed_random_indices(example, num_ents)

        # try:
        for i, ent_idx in enumerate(random_ents):
            # randomize span label
            if self.data_args.random_label:
                new_label = random.choice(span_labels)
                random_example.entities[ent_idx].type.natural = new_label
            
            # # randomize span indices
            # if self.data_args.random_span or self.data_args.perturb_span or self.data_args.mixed_random_span:
            #     random_example.entities[ent_idx].start = random_index_pairs[i][0]
            #     random_example.entities[ent_idx].end = random_index_pairs[i][1]
        # except Exception as e:
        #     print(e)
        #     import pdb; pdb.set_trace()

        return random_example

    def get_icl_examples(self, n=None, max_len_per_ex=None, legal_labels=None) -> List[Tuple[InputExample, str]]:
            """
            This formats the data in the dataset into the form needed for in-context learning, i.e.:

            Input: <input_sentence>
            Output: <output_sentence>

            It also includes the original InputExample itself, i.e. (InputExample, "Input: <input>\nOutput: <output>")
            """
        
            examples = []
            
            if self.is_classification_dataset: 
                idx = self.get_example_indices(n, cover_label_space=True)
            else:
                idx = self.get_example_indices(n)

            # retrieve examples
            for current_id in idx:
                example = self.get_example(current_id)
                input_str = self.input_format.format_input(example)

                # if random label
                if self.data_args.random_label and self.mode == "train":
                    example_to_format = self.randomize_example(
                        example,
                        list(self.natural_entity_types.values()),
                    )
                else:
                    example_to_format = example

                # format output normally
                correct_output = self.output_format.format_output(example_to_format)
                examples.append((example_to_format, input_str + correct_output))
            
    
            return examples
            

    def data_dir(self):
        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name)
        else:
            return os.path.join(self.data_path, self.name)


    def load_cached_data(self, cached_data_file: str):
        d = torch.load(cached_data_file)
        self.examples, self.features = d['examples'], d['features']


    def save_data(self, cached_data_file: str):
        torch.save({
            'examples': self.examples,
            'features': self.features,
        }, cached_data_file)


    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass


    @abstractmethod
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass


    def load_data(self, mode: str, seed: int = None) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split, seed=seed)

        return examples


    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )


    # def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
    #     input_sentences = [self.input_format.format_input(example, multitask=multitask, prompt_input=self.data_args.prompt_input) for example in self.examples]
    #     output_sentences = [self.output_format.format_output(example) for example in self.examples]
        
    #     if self.causal_lm:
    #         input_sentences = ["Input: " + input_sentences[i] + "\nOutput: " + output_sentences[i] + self.tokenizer.eos_token for i in range(len(input_sentences))]
    #         output_sentences = input_sentences

    #     input_tok = self.tokenizer.batch_encode_plus(
    #         input_sentences,
    #         max_length=max_input_length,
    #         return_tensors='pt',
    #         padding='max_length',
    #         truncation=True,
    #     )

    #     self._warn_max_sequence_length(max_input_length, input_sentences, "input")
    #     output_tok = self.tokenizer.batch_encode_plus(
    #         output_sentences,
    #         max_length=max_output_length,
    #         return_tensors='pt',
    #         padding='max_length',
    #         truncation=True,
    #     )

    #     if self.causal_lm:
    #         logging.warning("Ignoring pad tokens.")
    #         output_tok["input_ids"][output_tok["input_ids"] == self.tokenizer.pad_token_id] = -100

    #     self._warn_max_sequence_length(max_output_length, output_sentences, "output")

    #     assert input_tok.input_ids.size(0) == output_tok.input_ids.size(0)
        
    
    #     features = []

    #     for sentence_input_ids, att_mask, label_input_ids in zip(input_tok.input_ids, input_tok.attention_mask,
    #                                                              output_tok.input_ids):
    #         features.append(InputFeatures(
    #             input_ids=sentence_input_ids.tolist(),
    #             attention_mask=att_mask.tolist(),
    #             label_ids=label_input_ids.tolist(),
    #         ))
    
    #     return features


    # def generate_output_sentences(self, data_args: DataTrainingArguments, model, device, batch_size: int, cd_arg_dict=None) \
    #         -> Generator[Tuple[InputExample, str], None, None]:
    #     """
    #     Generate pairs (example, output_sentence) for evaluation.
    #     """
    #     test_data_loader = DataLoader(
    #         self,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         collate_fn=default_data_collator,
    #     )
    
    #     if self.causal_lm:
    #         max_length=data_args.max_output_seq_length_eval*2

    #     else:
    #         max_length=data_args.max_output_seq_length_eval

    #     for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
    #         if self.causal_lm:
    #             assert batch_size == 1
    #             input_toks = self.tokenizer.encode(
    #                             " ".join(["Input:"] + self.get_example(i).tokens), 
    #                             return_tensors='pt',
    #                         )
    #         else:
    #             input_toks = inputs['input_ids']

    #         if data_args.constrained_decoding:
    #             predictions = model.generate(
    #                 input_toks.to(device),
    #                 max_length=max_length,
    #                 num_beams=data_args.num_beams,
    #                 )
                                                
    #         else:

    #             predictions = model.generate(
    #                 input_toks.to(device),
    #                 max_length=max_length, 
    #                 num_return_sequences=1, 
    #                 # num_beams=data_args.num_beams,
    #                 do_sample=True,
    #                 top_k=50,
    #         )
        
            
    #         for j, (input_ids, label_ids, prediction) in enumerate(
    #                 zip(inputs['input_ids'], inputs['labels'], predictions)):
    #             current_id = i * batch_size + j
    #             example = self.get_example(current_id)
    #             output_sentence = self.tokenizer.decode(prediction, skip_special_tokens=True, #TODO: changed from true to false
    #                                                     clean_up_tokenization_spaces=False)

    #             if self.causal_lm:
    #                 try:
    #                     output_sentence = output_sentence.split("\n")[1][len("Output: "):]
    #                 except:
    #                     output_sentence = output_sentence

    #             yield example, output_sentence
            

    # def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False, tokenizer=None) \
    #         -> Dict[str, float]:
    #     """
    #     Evaluate model on this dataset.
    #     """

    #     # collect output to save
    #     example_ids = []
    #     output_sentences = []

    #     count = 0
    #     for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size, cd_arg_dict=None):
            
    #         if count <= 5:
    #             logging.warning("Input: {}".format(self.input_format.format_input(example, prompt_input=self.data_args.prompt_input)))
    #             logging.warning("Correct Output: {}".format(self.output_format.format_output(example)))
    #             logging.warning("Output: {}".format(output_sentence))
    #             count += 1

    #         example_ids.append(example.id)
    #         output_sentences.append(output_sentence)
    #         count += 1

    #         if count > self.data_args.val_examples:
    #             break

    #     assert len(example_ids) == len(output_sentences)
    #     return {"example_id": example_ids, "output_sentences": output_sentences}
    
    def _save_hf_data(self):
        raise NotImplementedError