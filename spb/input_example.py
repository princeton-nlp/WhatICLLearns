from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union
from torch.utils.data.dataset import Dataset
from copy import deepcopy
from spb.utils import augment_sentence, format_icl_output


@dataclass
class EntityType:
    """
    An entity type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences
    fine: str = None 
        
    def __hash__(self):
        return hash(self.short)


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    start: int                  # start index in the sentence
    end: int                    # end index in the sentence
    type: Optional[EntityType] = None   # entity type
    id: Optional[int] = None    # id in the current training/test example

    def to_tuple(self, fine_grained=False):
        if fine_grained:
            return self.type.fine, self.start, self.end
        else:
            return self.type.natural, self.start, self.end

    def __hash__(self):
        return hash((self.id, self.start, self.end))

@dataclass
class InputExample:
    """
    A single training/test example.
    """
    id: str                                      # unique id in the dataset
    tokens: List[str] = None                     # list of tokens (words)
    gt_tokens: Optional[List[str]] = None        # list of ground truth tokens
    dataset: Optional[Dataset] = None            # dataset this example belongs to

    # nli/multi-sentence
    sentence1: str = None
    sentence2: str = None
    
    # entity-relation extraction
    entities: List[Entity] = None                # list of entities
    # num_spans: Optional[int] = None
    
    # classification
    gt_label: int = None
    gt_label_name: str = None                # string version of label name
    orig_gt_label_name: str = None           # string version of label name

    # ICL experiments
    mapped_labels = False

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    label_ids: Optional[List[int]] = None
