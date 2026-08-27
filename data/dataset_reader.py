from wikipedia2vec.dump_db import DumpDB
import os
import pickle as pkl
from collections import OrderedDict
import json


KILT_DATASETS = OrderedDict(
    [
        # ("blink-train", "blink-train-kilt.jsonl"),
        # ("blink-dev", "blink-dev-kilt.jsonl"),
        ("aida-dev", "aida-dev-kilt.jsonl"),
        ("aida-test", "aida-test-kilt.jsonl"),
        ("msnbc-test", "msnbc-test-kilt.jsonl"),
        ("aquaint-test", "aquaint-test-kilt.jsonl"),
        ("ace2004-test", "ace2004-test-kilt.jsonl"),
        ("clueweb-test", "clueweb-test-kilt.jsonl"),
        ("wiki-test", "wiki-test-kilt.jsonl"),
        ("aida-train", "aida-train-kilt.jsonl"),
    ]
)


class KILTDatasetReader:
    def __init__(self, args, data_filename):
        self.args = args

        self.dump_db = DumpDB(os.path.join(args.wikipedia_path, args.dump_db_file))

        self.examples = self._create_examples(os.path.join(args.data_path, data_filename))

    def _create_examples(self, data_filename):
        examples = []
        for line in open(data_filename):
            example_dict = json.loads(line.strip())
            if self.args.second_step:
                gold_entity = example_dict["gold_entity"]
                candidates = example_dict['paired_candidates']  # not sorted by probs

            else:
                gold_entity = example_dict["output"][0]["answer"]
                try:
                    gold_entity = self.dump_db.resolve_redirect(gold_entity)
                except:
                    pass
                candidates = example_dict['candidates']
                candidates = [self.dump_db.resolve_redirect(c) for c in candidates]
                candidates = list(dict.fromkeys(candidates))
            
            example_id = example_dict['id']
            left_context = example_dict['meta']["left_context"]
            right_context = example_dict['meta']["right_context"]
            mention_word = example_dict['meta']["mention"]
            
            examples.append((example_id, left_context, right_context, mention_word, gold_entity, candidates, example_dict))
            
        return examples

