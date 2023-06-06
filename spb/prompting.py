
import logging
import random
import os
import copy
import openai

class Prompt():
    def __init__(
        self,
        data_args,
        demonstrations, # list of InputExample
        tokenizer,
        train_dataset,
        blurb="",
    ):        
        self.data_args = data_args
        self.tokenizer = tokenizer 
        self.train_dataset = train_dataset
        self.blurb = blurb
        self.demonstrations = demonstrations
        
        self.entity_types = self.train_dataset.natural_entity_types.copy()  # maps from label id to label str
        self.orig_entity_types = None                                       # maps from label id to orig label str
                                                                            # (label mapping only)
        self.prompt = None
        self.modified_label_space = False

        if self.train_dataset.is_classification_dataset:            
            # New label spaces: e.g. numbers, LMBFF labels
            if self.data_args.label_space is not None:
                self.entity_types = self.train_dataset.change_label_space(entity_type_dict=self.entity_types)
                self.modified_label_space = True

            if self.modified_label_space:
                self.demonstrations = [self.modify_labels(d) for d in demonstrations]
        

    def modify_labels(self, example):
        """
        Applies label modification to an example.
        """
        mod_example = copy.copy(example)
        if self.modified_label_space:
            mod_example.gt_label_name = self.entity_types[mod_example.gt_label]
            
        if self.data_args.replace_labels is not None:
            mod_example.orig_gt_label_name = self.orig_entity_types[mod_example.gt_label] 
              
        return mod_example    

    def get_prompt(self, example):
        """
        Returns modified test example + a prompt string to feed into the model.
        """
        if self.modified_label_space:
            example = self.modify_labels(example)
            
        input_str = self.format_example(example)[0]
        demo_strs = ["".join(self.format_example(ex)) for ex in self.demonstrations]
        return example, self.shorten_prompt(demo_strs, input_str).strip()
                
    def shorten_prompt(self, demonstrations, input_str):
        """
        Reduce a list of examples that are too long to one that fits the context window and return it as the prompt.
        """
        j = len(demonstrations)
        demo_sep = "".join(["\n" for i in range(self.data_args.demo_sep_lines)])

        toks = []
        while j == len(demonstrations) or len(toks[0]) > (self.data_args.max_seq_length_eval-self.data_args.max_new_tokens):
            train_examples = demo_sep.join([demonstrations[t] for t in sorted(random.sample(range(len(demonstrations)), j))])
            prompt = demo_sep.join([self.blurb, train_examples, input_str])
                
            toks = self.tokenizer.encode(
                    prompt, 
                    return_tensors='pt',
                    )

            j -= 1
        
        if j < len(demonstrations)-1:
            logging.warning(f"Reduced number of train examples to {j}")

        return prompt

    def format_example(self, example):
        return self.train_dataset.input_format.format_input(example), \
               self.train_dataset.output_format.format_output(example)


class ICLPromptHelper():
    """
    Utilities for making prompt generation easy.
    """
    
    def __init__(
        self,
        data_args,
        tokenizer,
        train_dataset,
        blurb="",
        gpt3_model_name=None,
        model=None,
    ):
        
        self.data_args = data_args
        self.tokenizer = tokenizer 
        self.train_dataset = train_dataset
        self.blurb = blurb
        self.newline_token_id = self.tokenizer.encode("\n")[-1]

        assert model == None or gpt3_model_name == None
        
        if model == None:
            # for querying OpenAI
            self.engine = self.get_correct_openai_engine(gpt3_model_name)           
            self.api_key = os.environ.get(f'{self.data_args.api_key_name}_openai')
        else:
            self.model = model

    
    def get_correct_openai_engine(self, model):
        """
        Gets name of OpenAI engine  from model_args.model_name_and_path.
        """
        _, engine = model.split("-")
        assert engine in ["ada", "babbage", "curie", "davinci"]
        return engine

    def get_prompt(self):
        train_example_list = self.train_dataset.get_icl_examples(
                self.data_args.num_prompt_ex,
                max_len_per_ex = int(self.data_args.max_seq_length / self.data_args.num_prompt_ex),
            )
        
        return Prompt(
            self.data_args,
            [t[0] for t in train_example_list], # list of InputExample
            self.tokenizer,
            self.train_dataset,
            blurb=self.blurb,
        )
        
        
    def format_example(self, example):
        return self.train_dataset.input_format.format_input(example), \
               self.train_dataset.output_format.format_output(example)
               
    def get_openai_response(self, prompt: Prompt, example, single_response=False):
        """ 
        Call OpenAI API for Codex or GPT3 usage.
        """

        example, prompt_str = prompt.get_prompt(example)
    
        if single_response:
            n = 1
        else:
            n = self.data_args.num_sampled_responses
        
        assert type(n) == int

        # set up openai api access
        openai.api_key = self.api_key
        
        if self.data_args.do_sample:
            temperature = 0.7
        else:
            temperature = 0

        if self.data_args.constrained_decoding:       
            labels = [self.tokenizer.encode(" " + label)[0] for label in prompt.entity_types.values()]
            max_tokens = 1
            logit_bias = {
                t: 100 for t in labels
            }
            
            responses = openai.Completion.create(
                engine=self.engine, 
                prompt=prompt_str, 
                max_tokens=max_tokens,
                n=n, 
                logit_bias=logit_bias,
                logprobs=5,
                temperature=temperature,
            )
            
        else:
            responses = openai.Completion.create(
                engine=self.engine, 
                prompt=prompt_str, 
                max_tokens=self.data_args.max_new_tokens, 
                n=n, 
                temperature=temperature
            )
                    
        if single_response:
            return example, prompt_str, responses.choices[0]["text"].split("\n")[0].strip()
        
        else:
            return example, prompt_str, [responses.choices[i]["text"] for i in range(n)]

    def clean_response(self, response):
        try:
            return response.strip().split("\n")[0].strip()
        except:
            return response

    

    def get_causal_lm_response(self, prompt: Prompt, example, single_response=False):
        
        example, prompt_str = prompt.get_prompt(example)

        toks = self.tokenizer(
                prompt_str,                                            
                max_length=self.data_args.max_seq_length_eval-self.data_args.max_new_tokens, 
                return_tensors='pt',
            )
        
        # Get constrained decoding
        newline_token_id = self.tokenizer.encode("\n")[-1]
        label_ids = [self.tokenizer.encode(" " + label)[-1] for label in prompt.entity_types.values()]

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            return label_ids
        
        # import pdb; pdb.set_trace()
        model_outputs = self.model.generate(
            toks["input_ids"].cuda(),
            attention_mask=toks["attention_mask"].cuda(),
            max_new_tokens=1,  
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            eos_token_id=newline_token_id,           
            do_sample=self.data_args.do_sample,   
            num_beams=self.data_args.num_beams,                     
        )

        # time.sleep(3)
        decoded_outputs = self.tokenizer.batch_decode(
            model_outputs, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        full_response = decoded_outputs.lstrip(self.tokenizer.pad_token)
        try:
            # If multiple extra lines were printed, only take the first one
            response = full_response[len(prompt_str):].split("\n")[0]
        
        except:
            response = full_response
        
        # Clean up any extra spaces at the start, if necessary
        if len(response) == 0:
            response = full_response

        if response[0] == " ":
            if len(response) == 1:
                response = full_response
            else:
                response = response[1:]

        return example, prompt_str, [response]
    


