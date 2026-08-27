from torch.utils.data import Dataset
import torch
from typing import List, Dict
from data.data_utils import read_instances
import random
from transformers import AutoTokenizer

class EntityDisambiguationDataset(Dataset):
    def __init__(self, args, setname, epoch=-1):

        self.args = args

        self.data = read_instances(args.data_path, setname, epoch=epoch)

        self.mask_mention = False
        if setname.find("train") != -1 and args.mask_mention:
            self.mask_mention = True

        self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        example_id = item['example_id']
        example_dict = item['example_dict']

        input_ids, token_type_ids, attention_mask = item['input_ids'], item['token_type_ids'], item['attention_mask']

        mention_spans = item['mention_spans']

        if self.mask_mention and random.random() < self.args.mask_mention_prob:
            mention_start_idx, mention_end_idx = mention_spans
            span_length = mention_end_idx - mention_start_idx + 1
            input_ids[mention_start_idx:mention_end_idx + 1] = [self.tokenizer.mask_token_id] * span_length
        
        candidates = item['candidates']
        candidates_spans = item['candidates_spans']
        candidates_mask = item['candidates_mask']

        num_candidates = sum(candidates_mask)
        global_attention_mask = [0] * len(input_ids)

        if self.args.global_mode == 1:
            global_attention_mask[0] = 1

        elif self.args.global_mode == 2:
            global_attention_mask[0] = 1

            m_start, m_end = mention_spans

            global_attention_mask[m_start] = 1

        elif self.args.global_mode == 3:
            global_attention_mask[0] = 1

            num_candidates = sum(candidates_mask)
            for s, e in candidates_spans[:num_candidates]:
                global_attention_mask[s] = 1

        elif self.args.global_mode == 4:
            global_attention_mask[0] = 1

            m_start, m_end = mention_spans
            global_attention_mask[m_start] = 1

            num_candidates = sum(candidates_mask)
            for s, e in candidates_spans[:num_candidates]:
                global_attention_mask[s] = 1
        else:
            print("global_option_error!!!!!")
            exit()
            
        gold_entity_position = item['gold_entity_position']
        gold_entity = item['gold_entity']

        fields = {"input_ids": torch.LongTensor(input_ids),
                  "token_type_ids": torch.LongTensor(token_type_ids),
                  "attention_mask": torch.LongTensor(attention_mask),
                  "global_attention_mask": torch.LongTensor(global_attention_mask),
                  "mention_spans": torch.LongTensor(mention_spans),
                  "candidates_spans": torch.LongTensor(candidates_spans),
                  "candidates_mask": torch.LongTensor(candidates_mask),

                  "gold_entity_positions": gold_entity_position,

                  # eval
                  "meta_data": {"example_id": example_id,
                                "gold_entity": gold_entity,
                                "candidates": candidates,
                                "example_dict": example_dict}
                  }
        
        return fields


def collate_ed_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    input_ids  [max_seq_len]
    token_type_ids  [max_seq_len]
    attention_mask  [max_seq_len]
    
    global_attention_mask  [max_seq_len]

    mention_spans  [2]
    candidates_spans  [max_num_candidates, 2]
    candidates_mask  [max_num_candidates]

    gold_entity_positions [1]

    meta_data : [example_id, gold_entity, candidates, example_dict]
    """

    batch_size = len(batch)
    outputs = {}

    fields = list(batch[0].keys())
    
    meta_fields = batch[0]["meta_data"].keys()
    outputs["meta_data"] = {field: [x["meta_data"][field] for x in batch] for field in meta_fields}

    fields.remove("meta_data")

    for field in fields:
        output = [batch[sample_idx][field] for sample_idx in range(batch_size)]
        output = torch.stack(output, dim=0) if field != "gold_entity_positions" else torch.tensor(output)
        outputs[field] = output
    
    return outputs
