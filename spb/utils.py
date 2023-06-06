# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Tuple, List, Dict
import re
import math
import numpy as np
import string
import time
import os
import openai
import random
import Levenshtein
from urllib.parse import non_hierarchical


#######################################################################
#
# Original utils written for TANL
#
#######################################################################

def get_episode_indices(episodes_string: str) -> List[int]:
    """
    Parse a string such as '2' or '1-5' into a list of integers such as [2] or [1, 2, 3, 4, 5].
    """
    episode_indices = []

    if episodes_string != None and episodes_string != '':
        ll = [int(item) for item in episodes_string.split('-')]

        if len(ll) == 1:
            episode_indices = ll

        else:
            _start, _end = ll
            episode_indices = list(range(_start, _end + 1))

    return episode_indices


def expand_tokens(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]],
                  entity_tree: Dict[int, List[int]], root: int,
                  begin_entity_token: str, sep_token: str, er_sep_token: str, relation_sep_token: str, end_entity_token: str, 
                  entity_only = False, before=False, entity_types=["location", "organization", "person", "other"]) \
        -> List[str]:
    """
    Recursively expand the tokens to obtain a sentence in augmented natural language.

    Used in the augment_sentence function below (see the documentation there).
    """
    new_tokens = []
    root_start, root_end = augmentations[root][1:] if root >= 0 else (0, len(tokens))
    i = root_start  # current index

    for entity_index in entity_tree[root]:
        tags, start, end = augmentations[entity_index]

        # add tokens before this entity
        new_tokens += tokens[i:start]

        # expand this entity
        new_tokens.append(begin_entity_token)
        new_tokens += expand_tokens(
            tokens, 
            augmentations, 
            entity_tree, 
            entity_index,
            begin_entity_token, sep_token, er_sep_token, relation_sep_token, end_entity_token, 
            entity_only, 
            before, 
            entity_types,
        )
        
        for tag in tags:
            # if tag[0] in entity_types:
            # import pdb; pdb.set_trace()
            if tag[0] and len(tag) == 1:
                # only append tag[0] if it is a type, otherwise skip the type
                new_tokens.append(sep_token)
                new_tokens.append(tag[0])


            else:
                if not entity_only:
                    if tag[0]:
                        new_tokens.append(er_sep_token)
                        new_tokens.append(tag[0])

                    if not entity_only:
                        for x in tag[1:]:
                            new_tokens.append(relation_sep_token)
                            new_tokens.append(x)

        new_tokens.append(end_entity_token)
        i = end

    # add tokens after all entities
    new_tokens += tokens[i:root_end]

    return new_tokens

def augment_sentence(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]], begin_entity_token: str,
                     sep_token: str, er_sep_token: str, relation_sep_token: str, end_entity_token: str, 
                     entity_only=False, before=False, entity_types=["location", "organization", "person", "other"]) -> str:
    """
    Augment a sentence by adding tags in the specified positions.

    Args:
        tokens: Tokens of the sentence to augment.
        augmentations: List of tuples (tags, start, end).
        begin_entity_token: Beginning token for an entity, e.g. '['
        sep_token: Separator token, e.g. '|'
        relation_sep_token: Separator token for relations, e.g. '='
        end_entity_token: End token for an entity e.g. ']'

    An example follows.

    tokens:
    ['Tolkien', 'was', 'born', 'here']

    augmentations:
    [
        ([('person',), ('born in', 'here')], 0, 1),
        ([('location',)], 3, 4),
    ]

    output augmented sentence:
    [ Tolkien | person | born in = here ] was born [ here | location ]
    """
    # sort entities by start position, longer entities first
    augmentations = list(sorted(augmentations, key=lambda z: (z[1], -z[2])))

    # check that the entities have a tree structure (if two entities overlap, then one is contained in
    # the other), and build the entity tree
    root = -1   # each node is represented by its position in the list of augmentations, except that the root is -1
    entity_tree = {root: []}        # list of children of each node
    current_stack = [root]          # where we are in the tree

    for j, x in enumerate(augmentations):
        tags, start, end = x
        if any(augmentations[k][1] < start < augmentations[k][2] < end for k in current_stack):
            # tree structure is not satisfied!
            logging.warning(f'Tree structure is not satisfied! Dropping annotation {x}')
            continue

        while current_stack[-1] >= 0 and \
                not (augmentations[current_stack[-1]][1] <= start <= end <= augmentations[current_stack[-1]][2]):
            current_stack.pop()

        # add as a child of its father
        entity_tree[current_stack[-1]].append(j)

        # update stack
        current_stack.append(j)

        # create empty list of children for this new node
        entity_tree[j] = []

    sentence = ' '.join(expand_tokens(
        tokens, 
        augmentations, 
        entity_tree, 
        root, 
        begin_entity_token, sep_token, er_sep_token, relation_sep_token, end_entity_token,
        entity_only, 
        before, 
        entity_types,
    ))

    if before:
        return flip_order(sentence, begin_entity_token, sep_token, er_sep_token, relation_sep_token, end_entity_token)

    else:
        return sentence

def get_span(l: List[str], span: List[int]):
    assert len(span) == 2
    return " ".join([l[i] for i in range(span[0], span[1]) if i < len(l)])


def get_precision_recall_f1(num_correct, num_predicted, num_gt):
    assert 0 <= num_correct <= num_predicted
    assert 0 <= num_correct <= num_gt

    precision = num_correct / num_predicted if num_predicted > 0 else 0.
    recall = num_correct / num_gt if num_gt > 0 else 0.
    f1 = 2. / (1. / precision + 1. / recall) if num_correct > 0 else 0.

    return precision, recall, f1


########################################################################
#
# Utils written for SPB (warning, it gets uglier the further you scroll)
#
#######################################################################
def add_period_for_lmbff(sentence):
        sentence = sentence.rstrip()
        if sentence[-1] not in string.punctuation:
            sentence = sentence.rstrip() + "."
        return sentence

def randomize_list_without_repeats(orig_list):
    random.seed()
    shuffled_list = list(orig_list).copy()
    while True:
        random.shuffle(shuffled_list)
        for a, b in zip(orig_list, shuffled_list):
            if a == b:
                break
        else:
            return shuffled_list


def remove_endpoint_duplicates(l, first=True):
    if first:
        while l.count(l[1]) > 1:
            l[1] = l[1] + 1
    else:
        while l.count(l[-2]) > 1:
            l[-2] = l[-2] - 1
    
    return l

def add_punctuation_spaces(sent):
    """
    For data that doesn't have a space before punctuation, add it in.
    """
    extra_spaces = 0
    for p in string.punctuation:
        sent = sent.replace(p, f" {p}")
        extra_spaces += sent.count(p)

    new_sent = ""
    for i, s in enumerate(sent):

        # If s is punctuation and there's no space after it
        if (
            s in string.punctuation 
            and i < len(sent)-1
            and sent[i+1] != " "
        ):
            new_sent += f"{s} "
            extra_spaces += 1

        else:
            new_sent += s

    return new_sent, extra_spaces

def split_icl_example(clean_example, link_str="\nOutput: "):
    if link_str is None:
        link_str = "\nOutput: "
    return clean_example.split(link_str)
    
def format_icl_output(s, input_format)-> str:
    """
    Perform simple formatting of ICL output, given input and output strings.
    """
    if input_format in ["complex_foo_input", "foo_input"]:
        link_str = "\nassert ans == "
        input_str, output_str = s.split(link_str)

        return input_str + link_str, output_str
        
    else:
        input_str, output_str = s.split("\nOutput: ")
        return "\nOutput: ".format(input_str, output_str) 


def get_device_map(model_config, n_devices: int)-> dict:
    """
    Returns device map for model parallelism.
    """

    if hasattr(model_config, "num_layers"):
        n_layers = model_config.num_layers
    elif hasattr(model_config, "n_layer"):
        n_layers = model_config.n_layer
    elif hasattr(model_config, "n_layers"):
        n_layers = model_config.n_layers
    elif hasattr(model_config, "num_hidden_layers"):
        n_layers = model_config.num_hidden_layers
    else:
        assert(n_layers is not None)

    device_map = {
                    i: range(i*math.floor(n_layers/n_devices), (i+1)*math.floor(n_layers/n_devices)) for i in range(n_devices)
                }

    if n_layers % n_devices != 0:
        device_map[n_devices-1] = range(device_map[n_devices-1][0], n_layers)

    return device_map

# def get_special_toks(example, input_seq: str , gt_output_seq: str) -> List[str]:
#     """
#     Quick proxy to calculate how many special tokens are in the ground truth output of a dataset.
#     """
    
#     orig_tokens = example.tokens
#     special_toks = []

#     # stem original tokens
#     snowball = SnowballStemmer(language='english')
#     stemmed_orig_tokens = [snowball.stem(tok) for tok in orig_tokens]

#     for tok in input_seq.split(" ") + gt_output_seq.split(" "):
#         if tok not in orig_tokens and snowball.stem(tok) not in stemmed_orig_tokens:
#             special_toks.append(tok)
    
#     return special_toks

def get_edit_distance(correct_sentence: str, output_sentence: str) -> float:
    """
    Returns inverse normalized edit distance (higher is better!!!!)
    """

    d = Levenshtein.distance(correct_sentence, output_sentence)
    m = max(len(correct_sentence), len(output_sentence))
    return round((m-d)/m, 3)

def clean_special_tokens(output_sentence):
    if "<pad>" == output_sentence[:5]:
        start = len("<pad>")+1
    else:
        start = 0

    if "</s>" in output_sentence:
        end = output_sentence.index("</s>")
    else:
        end = len(output_sentence)
    return output_sentence[start:end].strip()

def get_ents_in_rel(relations, relation_types):
    """
    From a relation string "<head_ent> <relation type> <tail_ent>", return (<head_ent>, <tail_ent>)
    """
    rel_tuples = []

    relation_types = set(relation_type.natural for relation_type in relation_types.values())
    for rel in relations:
        for rt in relation_types:
            if rt in rel:
                rel_tuples.append(rel.split(rt))
    
    return [item for sublist in rel_tuples for item in sublist]
    


def is_matched(expression, opening, closing):
    """
    Finds out how balanced an expression is.
    With a string containing only brackets.

    >>> is_matched('[]()()(((([])))')
    False
    >>> is_matched('[](){{{[]}}}')
    True
    """
    queue = []

    for letter in expression:
        if letter == opening:
            queue.append(closing)
        elif letter in closing:
            if not queue or letter != queue.pop():
                return False
    return not queue


def check_output_format(sentence, begin_entity_token: str, sep_token: str, er_sep_token: str, relation_sep_token: str, end_entity_token: str):
    ent_starts = list(re.finditer("\{}".format(begin_entity_token), sentence))
    ent_ends = list(re.finditer("\{}".format(end_entity_token), sentence))
    ent_seps = list(re.finditer("\{}".format(sep_token), sentence))

    if(len(ent_ends) != len(ent_starts)) or (len(ent_seps) != len(ent_ends)):
        return None
        
    elif not is_matched("".join([x for x in sentence if x in [begin_entity_token, end_entity_token]]), begin_entity_token, end_entity_token):
        # print("here")
        # print(sentence)
        # print("".join([x for x in sentence if x in [begin_entity_token, end_entity_token]]))
        return None
    
    else:
        return ent_starts, ent_ends, ent_seps

def flip_order(sentence, begin_entity_token: str, sep_token: str, er_sep_token: str, relation_sep_token: str, end_entity_token: str):
    
    check_output = check_output_format(sentence, begin_entity_token, sep_token, er_sep_token, relation_sep_token, end_entity_token)

    if check_output is None:
        print(sentence)
        return "INCORRECT FORMAT"

    else:
        ent_starts, ent_ends, ent_seps = check_output

    corrected = []
    phrases = []
    for i in range(len(ent_ends)):
        phrase = sentence[ent_starts[i].end()+1:ent_ends[i].start()-1]
        split_phrase = [p.strip() for p in phrase.split(" {} ".format(sep_token))]

        if len(split_phrase) == 2:
            phrases.append(phrase)
            corrected.append(split_phrase[1] + " {} ".format(sep_token) + split_phrase[0])
    
    for i in range(len(ent_ends)):
        sentence = sentence.replace(phrases[i], corrected[i])
    
    print("Flipped: {}:".format(sentence))
    return sentence

