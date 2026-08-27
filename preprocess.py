import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import json
import copy
from collections import OrderedDict
from data.data_utils import load_wiki_title_info
from data.vocab import (
    additional_tokens,
    mention_start_token,
    mention_end_token,
    entity_end_token,
    title_end_token,
    description_end_token,
    type_sep_token,
    empty_description_token,
    empty_type_token,
    allowed_relations,
)

from data.data_utils import load_all_examples
import numpy as np
import torch

TRIMMING_STATS = {}



def get_entity_tokens(args, tokenizer, wiki_title_info, title):
    if wiki_title_info is not None and title in wiki_title_info:
        description, triples = wiki_title_info[title]
    else:
        description, triples = None, None

    title_tokens = tokenizer.tokenize(title)
    title_len = len(title_tokens)

    if args.add_desc or args.add_type:
        entity_tokens = title_tokens + [title_end_token]
    else:
        entity_tokens = title_tokens

    if args.add_desc:
        if description is not None:
            entity_tokens += tokenizer.tokenize(description)
        else:
            entity_tokens += [empty_description_token]

    if args.add_desc and args.add_type:    
        entity_tokens += [description_end_token]

    if args.add_type:
        if triples is not None:
            type_tokens = []
            for rel, objects in triples.items():
                if rel in allowed_relations:
                    rel_obj_str = f"{rel}: {', '.join(objects[:2])}"
                    type_tokens += tokenizer.tokenize(rel_obj_str) + [type_sep_token]

            if type_tokens:
                entity_tokens += type_tokens[:-1]  # remove final [type_sep_token]
            else:
                entity_tokens += [empty_type_token]

        else:
            entity_tokens += [empty_type_token]

    entity_tokens += [entity_end_token]

    return entity_tokens, title_len


def prune_candidates_and_generate_candidate_tokens(args, 
                                                   setname, 
                                                   tokenizer, 
                                                   max_length, 
                                                   total_candidates, 
                                                   gold_entity,
                                                   all_train_entities, 
                                                   wiki_title_info,
                                                   start=0, 
                                                   train=False):

    # total_candidates is copied before function call 
    max_length = max_length - 1  # remove [CLS]

    if train:
        # gold_entity_tokens = tokenizer.tokenize(gold_entity) + [entity_end_token]
        gold_entity_tokens, gold_title_len = get_entity_tokens(args, tokenizer, wiki_title_info, gold_entity)

        max_length -= len(gold_entity_tokens)

        if gold_entity in total_candidates:
            gold_entity_idx = total_candidates.index(gold_entity)
            total_candidates.remove(gold_entity)
        else:
            gold_entity_idx = -1

        sampled = False
        # opt == 1 sampling no, opt == 2 or opt == 3 sampling yes
        if (args.opt == 2 or args.opt == 3) and len(total_candidates) == 0:
            sample_size = random.randint(1, 99)

            filtered_train_entities = [entity for entity in all_train_entities if entity != gold_entity]

            total_candidates = random.sample(filtered_train_entities, sample_size)

            sampled = True

        if args.random_shuffle:
            random.shuffle(total_candidates)
    # =================================================================================================== #

    candidates = []
    nested_candidate_tokens = []
    title_lengths = []
    ent_seq_len = 0
    for entity in total_candidates:
        # entity_tokens = tokenizer.tokenize(entity) + [entity_end_token]
        entity_tokens, title_len = get_entity_tokens(args, tokenizer, wiki_title_info, entity)
        if ent_seq_len + len(entity_tokens) > max_length:
            break
            
        nested_candidate_tokens.append(entity_tokens)
        candidates.append(entity)
        title_lengths.append(title_len)
        ent_seq_len += len(entity_tokens)
    
    assert len(candidates) == len(nested_candidate_tokens)

    if train:
        if args.random_shuffle:
            idx = random.randint(0, len(candidates))
            candidates.insert(idx, gold_entity)
            nested_candidate_tokens.insert(idx, gold_entity_tokens)
            title_lengths.insert(idx, gold_title_len)
        else:
            # answer insert
            if sampled:
                if args.opt == 2:  # insert first position
                    candidates = [gold_entity] + candidates
                    nested_candidate_tokens = [gold_entity_tokens] + nested_candidate_tokens
                    title_lengths = [gold_title_len] + title_lengths
                elif args.opt == 3:  # insert random position
                    idx = random.randint(0, len(candidates))
                    candidates.insert(idx, gold_entity)
                    nested_candidate_tokens.insert(idx, gold_entity_tokens)
                    title_lengths.insert(idx, gold_title_len)
            else:
                if gold_entity_idx == -1:
                    candidates = candidates + [gold_entity]
                    nested_candidate_tokens = nested_candidate_tokens + [gold_entity_tokens]
                    title_lengths = title_lengths + [gold_title_len]
                else:
                    candidates.insert(gold_entity_idx, gold_entity)
                    nested_candidate_tokens.insert(gold_entity_idx, gold_entity_tokens)
                    title_lengths.insert(gold_entity_idx, gold_title_len)

        ent_seq_len += len(gold_entity_tokens)

    assert len(candidates) == len(nested_candidate_tokens)
      
    candidate_tokens = []
    candidate_spans = []
    for entity_tokens, title_len in zip(nested_candidate_tokens, title_lengths):
        candidate_tokens += entity_tokens
        candidate_spans.append((start, start + title_len - 1))
        
        start += len(entity_tokens)

    assert len(candidates) == len(nested_candidate_tokens) == len(candidate_spans)
    assert len(candidate_tokens) == ent_seq_len
    
    return candidate_tokens, candidates, candidate_spans


def second_step_prune_candidates_and_generate_candidate_tokens(args, 
                                                               setname, 
                                                               tokenizer, 
                                                               max_length, 
                                                               total_paired_candidates,
                                                               gold_entity, 
                                                               all_train_entities, 
                                                               wiki_title_info, 
                                                               start=0, 
                                                               train=False):
    max_length = max_length - 1  # remove [CLS]
    
    top_k = args.max_num_candidates # prune_candidates_and_generate_candidate_tokens

    # sort by priors in descending order
    total_paired_candidates = sorted(total_paired_candidates, key=lambda item: item[1], reverse=True)

    total_candidates = [c for c, p in total_paired_candidates]
    total_probs = [p for c, p in total_paired_candidates]

    if train:
        assert gold_entity in total_candidates

        sample_num = top_k - 1

        gold_entity_idx = total_candidates.index(gold_entity)
        gold_entity_prob = total_probs[gold_entity_idx]
        total_candidates.pop(gold_entity_idx)
        total_probs.pop(gold_entity_idx)
        gold_entity_tokens, gold_title_len = get_entity_tokens(args, tokenizer, wiki_title_info, gold_entity)
        max_length -= len(gold_entity_tokens)

        if len(total_candidates) <= sample_num:
            sampled_candidates = total_candidates
            sampled_candidate_probs = total_probs
        else:
            if args.sampling == "greedy":
                sampled_candidates = total_candidates[:sample_num]
                sampled_candidate_probs = total_probs[:sample_num]

            elif args.sampling == "random":
                sampled_indices = random.sample(range(len(total_candidates)), sample_num)
                sampled_candidates = [total_candidates[i] for i in sampled_indices]
                sampled_candidate_probs = [total_probs[i] for i in sampled_indices]

                # restore order
                tmp = list(zip(sampled_candidates, sampled_candidate_probs))
                tmp.sort(key=lambda x: x[1], reverse=True)
                sampled_candidates = [c for c, p in tmp]
                sampled_candidate_probs = [p for c, p in tmp]

            elif args.sampling == "probability":
                weights = np.array(total_probs) / sum(total_probs)
                sampled_indices = np.random.choice(len(total_candidates), sample_num, replace=False, p=weights)

                sampled_candidates = [total_candidates[i] for i in sampled_indices]
                sampled_candidate_probs = [total_probs[i] for i in sampled_indices]

                # restore order
                tmp = list(zip(sampled_candidates, sampled_candidate_probs))
                tmp.sort(key=lambda x: x[1], reverse=True)
                sampled_candidates = [c for c, p in tmp]
                sampled_candidate_probs = [p for c, p in tmp]

            else:
                print("must select sampling in ['greedy', 'random', 'probability']")
                exit()

        gold_entity_idx = sum(p > gold_entity_prob for p in sampled_candidate_probs)

    else:  # Eval
        sampled_candidates = total_candidates[:top_k]
        sampled_candidate_probs = total_probs[:top_k]

    candidates = []
    nested_candidate_tokens = []
    title_lengths = []
    ent_seq_len = 0
    for entity in sampled_candidates:
        entity_tokens, title_len = get_entity_tokens(args, tokenizer, wiki_title_info, entity)
        if ent_seq_len + len(entity_tokens) > max_length:
            break
            
        nested_candidate_tokens.append(entity_tokens)
        candidates.append(entity)
        title_lengths.append(title_len)
        ent_seq_len += len(entity_tokens)

    assert len(candidates) == len(nested_candidate_tokens)
    
    if train:
        gold_entity_idx = min(gold_entity_idx, len(candidates))  # if trimmed

    if train:
        candidates.insert(gold_entity_idx, gold_entity)
        nested_candidate_tokens.insert(gold_entity_idx, gold_entity_tokens)
        title_lengths.insert(gold_entity_idx, gold_title_len)
        ent_seq_len += len(gold_entity_tokens)

    assert len(candidates) == len(nested_candidate_tokens)
      
    candidate_tokens = []
    candidate_spans = []
    for entity_tokens, title_len in zip(nested_candidate_tokens, title_lengths):
        candidate_tokens += entity_tokens
        candidate_spans.append((start, start + title_len - 1))

        start += len(entity_tokens)

    assert len(candidates) == len(nested_candidate_tokens) == len(candidate_spans)
    assert len(candidate_tokens) == ent_seq_len
    
    # ====================================================== #
    TRIMMING_STATS[setname]["total_examples"] += 1
    TRIMMING_STATS[setname]["trimmed_examples"] += int(len(candidates) < len(sampled_candidates))
    TRIMMING_STATS[setname]["sum_original_k"] += len(sampled_candidates)
    TRIMMING_STATS[setname]["sum_survived_k"] += len(candidates)

    gold_trimmed = int((gold_entity in sampled_candidates) and
                       (gold_entity not in candidates)
                      )
    TRIMMING_STATS[setname]["gold_trimmed"] += gold_trimmed
    # ====================================================== #
    return candidate_tokens, candidates, candidate_spans


def example2instance(args, setname, example, tokenizer, all_train_entities, wiki_title_info, train=False):
    example_id, left_context, right_context, mention_word, gold_entity, total_candidates, example_dict = example

    max_seq_len = args.max_seq_len if train else args.eval_max_seq_len

    left_context_tokens = tokenizer.tokenize(left_context)  # tokenizer.tokenize(" " + left_context)
    mention_word_tokens = tokenizer.tokenize(" " + mention_word)
    right_context_tokens = tokenizer.tokenize(" " + right_context)

    mention_word_length = len(mention_word_tokens) + 2  # add </Mention>, <Mention>

    context_tokens = left_context_tokens + [mention_start_token] + mention_word_tokens + [mention_end_token] + right_context_tokens
    context_tokens = [tokenizer.cls_token] + context_tokens + [tokenizer.sep_token]
    # assert len(context_tokens) <= args.max_context_len

    context_token_ids = tokenizer.convert_tokens_to_ids(context_tokens)

    mention_start = len(left_context_tokens) + 2  # add [CLS], <Mention>
    mention_end = mention_start + mention_word_length - 1 - 2  # remove <Mention>, </Mention>

    mention_spans = [mention_start, mention_end]
    
    assert context_tokens[mention_start-1] == mention_start_token and context_tokens[mention_end+1] == mention_end_token
    
    # if train mode then random entity may be inserted in the total_candidates, so use copy
    max_length = max_seq_len - len(context_tokens)  # max_length : maximum candidates seq length

    total_candidates_copied = copy.deepcopy(total_candidates)
    if not args.second_step:  # first step
        candidate_tokens, candidates, candidates_spans \
        = prune_candidates_and_generate_candidate_tokens(args, setname, tokenizer, max_length, total_candidates_copied, gold_entity, 
                                                         all_train_entities, wiki_title_info,
                                                         start=len(context_token_ids), train=train)
    else:  # second step
        candidate_tokens, candidates, candidates_spans \
        = second_step_prune_candidates_and_generate_candidate_tokens(args, setname, tokenizer, max_length, total_candidates_copied, gold_entity, 
                                                                     all_train_entities, wiki_title_info,
                                                                     start=len(context_token_ids), train=train)

    candidate_tokens = candidate_tokens + [tokenizer.sep_token]
  
    candidate_token_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)

    assert len(candidate_token_ids) <= max_length
    
    gold_entity_position = candidates.index(gold_entity) if gold_entity in candidates else - 1
    
    assert not (train and gold_entity_position == -1), "Error: gold_entity_position should not be -1 when train=True"
    
    input_ids = context_token_ids + candidate_token_ids
    
    # sequence processing 
    seq_len = len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * (max_seq_len - seq_len)
    attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
    token_type_ids = [0] * len(input_ids)

    # candidate processing
    num_candidates = len(candidates)
    candidates_mask = [1] * num_candidates + [0] * (args.max_num_candidates - num_candidates)
    candidates_spans = candidates_spans + [(0, 0)] * (args.max_num_candidates - num_candidates)

    instance = {"example_id": example_id,
                "input_ids": input_ids,  # [max_seq_len]
                "token_type_ids": token_type_ids,  # [max_seq_len]
                "attention_mask": attention_mask,  # [max_seq_len]
                "mention_spans": mention_spans,  # [2]
                "total_candidates": total_candidates,
                "candidates": candidates,  # [num_candidates]
                "candidates_spans": candidates_spans,  # [max_num_candidates]
                "candidates_mask": candidates_mask,  # [max_num_candidates]
                "gold_entity_position": gold_entity_position,
                "gold_entity": gold_entity,
                "example_dict": example_dict, 
                }
    
    return instance
    

def get_all_candi_entities(examples):
    all_entities = []
    all_entities_set = set()
    for example in examples:
        example_id, left_context, right_context, mention_word, gold_entity, candidates, example_dict = example

        for c in candidates:
            if c not in all_entities_set:
                all_entities_set.add(c)
                all_entities.append(c)

    return all_entities

def save_instances(args, examples_dict):
    os.makedirs(args.save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokenizer.add_tokens(additional_tokens)
    
    wiki_title_info = None
    if args.second_step:
        wiki_title_info_path = os.path.join(args.wikipedia_path, args.wiki_title_info_file)
        wiki_title_info = load_wiki_title_info(wiki_title_info_path)
        
    train_examples = examples_dict['aida-train']

    if not args.second_step:
        all_train_entities = get_all_candi_entities(train_examples)
    else:
        all_train_entities = None

    example_num = OrderedDict()
    for setname, examples in examples_dict.items():
        example_num[setname] = 0

    for setname, examples in examples_dict.items():
        TRIMMING_STATS[setname] = {
            "total_examples": 0,
            "trimmed_examples": 0,
            "sum_original_k": 0,
            "sum_survived_k": 0,
            "gold_trimmed": 0,
        }

        if setname == "aida-train":
            for epoch in range(1, args.epochs + 1):
                with open(os.path.join(args.save_path, "epoch%d_%s_instances.json" % (epoch, setname)), "w") as f:
                    for ii, example in enumerate(tqdm(examples,
                                                               total=len(examples),
                                                               desc="epoch%d_%s" % (epoch, setname))):
                        instance = example2instance(args, setname, example, tokenizer, all_train_entities, wiki_title_info,
                                                    train=True)
                        if instance is not None: 
                            f.write(json.dumps(instance) + "\n")
                            if epoch == 1:
                                example_num[setname] += 1

        else:
            with open(os.path.join(args.save_path, "%s_instances.json" % setname), "w") as f:
                for ii, example in enumerate(tqdm(examples, total=len(examples), desc=setname)):
                    instance = example2instance(args, setname, example, tokenizer, all_train_entities, wiki_title_info,
                                                train=False)
                    if instance is not None: 
                        f.write(json.dumps(instance) + "\n")
                        example_num[setname] += 1

    meta_path = os.path.join(args.save_path, args.meta_file)
    with open(meta_path, "w") as of:
        of.write(json.dumps({"num_examples": example_num}))

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    examples_dict = load_all_examples(args)

    save_instances(args, examples_dict)

    if "aida-train" in TRIMMING_STATS:
        for key in ["total_examples", "trimmed_examples", "sum_original_k", "sum_survived_k", "gold_trimmed"]:
            if key in TRIMMING_STATS["aida-train"]:
                TRIMMING_STATS["aida-train"][key] = TRIMMING_STATS["aida-train"][key] // args.epochs

    trim_path = os.path.join(args.save_path, "trimming_stats.json")
    with open(trim_path, "w") as f:
        json.dump(TRIMMING_STATS, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=1013, help='')
    parser.add_argument('--data_path', type=str, default='./datasets', help='')
    parser.add_argument('--save_path', type=str, default='data', help='')

    parser.add_argument("--example_file", type=str, default="examples.pkl", help='')
    parser.add_argument("--meta_file", type=str, default="meta.json")

    parser.add_argument('--epochs', type=int, default=20, help='')

    parser.add_argument('--wikipedia_path', type=str, default='', help='')
    parser.add_argument("--dump_db_file", type=str, default="enwiki-latest.db", help='')
    parser.add_argument("--wiki_title_info_file", type=str, default="wiki_title_info.json", help='')

    parser.add_argument('--lm_name', type=str, default="allenai/longformer-large-4096",
                        choices=['allenai/longformer-base-4096', 'allenai/longformer-large-4096'], help='')

    parser.add_argument('--max_seq_len', type=int, default=1024, help='')
    parser.add_argument('--eval_max_seq_len', type=int, default=1024, help='')
    parser.add_argument('--max_context_len', type=int, default=256, help='')

    parser.add_argument('--max_num_candidates', type=int, default=100, help='')

    parser.add_argument('--opt', type=int, default=3, choices=[1, 2, 3], help='')

    parser.add_argument('--max_title_len', type=int, default=15, help='')

    parser.add_argument('--random_shuffle', help='', default=False, action='store_true')

    # second step
    parser.add_argument('--second_step', help='', default=False, action='store_true')

    parser.add_argument('--add_desc', help='', default=False, action='store_true')
    parser.add_argument('--add_type', help='', default=False, action='store_true')

    parser.add_argument('--sampling', type=str, default='greedy', choices=['greedy', 'random', 'probability'], help='')

    args = parser.parse_args()

    main(args)
