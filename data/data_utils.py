import os
import json
import pickle as pkl
from copy import deepcopy
from collections import OrderedDict
from data.dataset_reader import KILT_DATASETS, KILTDatasetReader
from tqdm import tqdm


def load_all_examples(args):
    # --------------------------------- Read dataset --------------------------------- #
    examples_dict = OrderedDict()
    for setname, data_filename in KILT_DATASETS.items():
        dataset_reader = KILTDatasetReader(args, data_filename)
        examples_dict[setname] = dataset_reader.examples
    
    return examples_dict


def read_instances(save_path, setname, epoch=-1):
    if setname.find("train") != -1:
        instance_file = "epoch%d_%s_instances.json" % (epoch, setname)
    else:
        instance_file = "%s_instances.json" % setname

    instance_path = os.path.join(save_path, instance_file)

    instances = [json.loads(line.strip()) for line in open(instance_path)]
    return instances


def load_titles(args):
    f = open(os.path.join(args.wikipedia_path, "wiki_titles.txt"))

    target_tiles = [title.strip() for  title in f]
    
    return target_tiles

def load_wiki_title_info(wiki_title_info_path):
    wiki_title_info = {}
    with open(wiki_title_info_path, encoding="utf-8") as f:

        for line in tqdm(f):
            item = json.loads(line.strip())
            title = item["title"]
            description = item["description"]
            triples = item["triples"]

            wiki_title_info[title] = (description, triples)

    return wiki_title_info
