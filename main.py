import argparse
from tqdm import tqdm
import numpy as np
import os
import random
import torch
import json
from model import DotProductScoringEDModel, CandidateClassificationEDModel, ExtractiveQAEDModel
import math
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler
from utils.logger_utils import set_logger
from utils.optimization_utils import create_optimizer, create_scheduler
from utils.evauation_utils import evaluate_ed_InKBF1
from data.entity_disambiguation_dataset import EntityDisambiguationDataset, collate_ed_data
import statistics
from distutils.util import strtobool

models = {'dotproduct': DotProductScoringEDModel,
          'classification': CandidateClassificationEDModel,
          'extractive': ExtractiveQAEDModel,
          }

def train(args):
    os.makedirs(args.save_path, exist_ok=True)
    
    log = set_logger(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = os.path.join(args.save_path, 'checkpoint_best.pt')
    
    log.info("Build Model......")
    model = models[args.model_type](args, device)

    # --------------------------------------------------
    # Load checkpoint for Step2 or warm-start training
    # --------------------------------------------------
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            print(f"[Loading checkpoint from {args.checkpoint}]")
            state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            new_state_dict = {}
            for key, weight in state_dict.items():
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = weight

            model.load_state_dict(new_state_dict, strict=False)
            print("[Checkpoint loaded successfully]")
        else:
            print(f"[Warning] checkpoint path does not exist: {args.checkpoint}")

    if args.eval:
        log.info('[Loading best model.....]')
        state_dict = torch.load(model_file, map_location=torch.device('cpu'),weights_only=True)

        new_state_dict = {}
        for key, weight in state_dict.items():
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = weight

        model.load_state_dict(new_state_dict)
        log.info('[Best Model Loaded!!!]')

    model.to(device)
    if args.compile:
        model = torch.compile(model)
    log.info("Build Model Complete!!!!")

    dev_dataset = EntityDisambiguationDataset(args, 'aida-dev')
    test_dataset = EntityDisambiguationDataset(args, 'aida-test')

    dev_data_loader = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size,
                                 shuffle=False, collate_fn=collate_ed_data)

    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size,
                                  shuffle=False, collate_fn=collate_ed_data)

    if args.eval:
        all_data_evaluate_and_save(args, device, model, log)
        exit()

    use_amp = args.amp
    scaler = GradScaler(enabled=use_amp)

    meta_file = os.path.join(args.data_path, "meta.json")
    meta = json.loads(open(meta_file).read())
    
    num_training_examples = meta["num_examples"]["aida-train"]
    
    num_training_batches = math.ceil(num_training_examples / args.train_batch_size)

    num_training_step = math.ceil(num_training_batches / args.gradient_accumulation_steps) * args.epochs

    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer, num_training_step)
    
    log.info("***** Running training *****")
    torch.cuda.empty_cache()

    epoch_0 = 1
    train_steps = 0
    train_loss = 0.0
    best_dev_f1 = 0.0
    patience = 0
    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.info('Epoch {}'.format(epoch))

        train_dataset = EntityDisambiguationDataset(args, 'aida-train', epoch)

        train_sampler = RandomSampler(train_dataset)  # if args.local_rank == -1 else DistributedSampler(train_dataset)

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                                       sampler=train_sampler, collate_fn=collate_ed_data)

        total_batch = len(train_data_loader)

        assert num_training_batches == total_batch
        
        model.train()
        with tqdm(total=total_batch) as pbar:
            for step, batch in enumerate(train_data_loader, start=1):
                del batch['meta_data']

                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                accumulation_factor = args.gradient_accumulation_steps
                remaining_batches = total_batch % args.gradient_accumulation_steps
                if remaining_batches > 0 and step > total_batch - remaining_batches:
                    accumulation_factor = remaining_batches

                with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**batch)

                loss = loss / accumulation_factor

                scaler.scale(loss).backward()

                if (step % args.gradient_accumulation_steps == 0 or step == total_batch):
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)

                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()
                    
                    optimizer.zero_grad()

                    train_steps += 1

                    mean_loss = train_loss / train_steps

                    if train_steps % 10 == 0:
                        pbar.set_description("Epoch = {}, train_steps = {}, loss = {:.5f}".format(epoch, train_steps, mean_loss))

                    if train_steps % (400 // args.gradient_accumulation_steps) == 0:
                        log.info("Epoch = {}, train_steps = {}, loss = {:.5f}".format(epoch, train_steps, mean_loss))

                train_loss += loss.item()

                pbar.update(1)

        log.info('Start Dev & Test Evaluation in Epoch %d' % epoch)

        dev_precision, dev_recall, dev_f1, _ = evaluate(args, device, model, dev_data_loader)
        log.info('[Epoch {0:2d}][{1:>10}] precision: {2:.2f}, recall: {3:.2f} , F1: {4:.2f}'.format(epoch, "aida-dev", dev_precision, dev_recall, dev_f1))

        test_precision, test_recall, test_f1, _ = evaluate(args, device, model, test_data_loader)
        log.info('[Epoch {0:2d}][{1:>10}] precision: {2:.2f}, recall: {3:.2f} , F1: {4:.2f}'.format(epoch, "aida-test", test_precision, test_recall, test_f1))
        
        if dev_f1 >= best_dev_f1:
            torch.save(model.state_dict(), model_file)
            best_dev_f1 = dev_f1
            patience = 0
            log.info('[new best model saved.]')
        else:
            patience += 1

        if patience >= args.max_patience:
            log.info('Early stopping at epoch {} !!!!'.format(epoch))
            break

def all_data_evaluate_and_save(args, device, model, log):
    log.info('Start Dev & Test Evaluation in Best Model ......')

    setnames = ['aida-train', 'aida-dev', 'aida-test', 'msnbc-test', 'aquaint-test','ace2004-test','clueweb-test', 'wiki-test']

    scores = []
    scores_oov = []
    for setname in setnames:
        if setname == 'aida-train':
            dataset = EntityDisambiguationDataset(args, setname, epoch=1)
        else:
            dataset = EntityDisambiguationDataset(args, setname)

        data_loader = DataLoader(dataset=dataset, batch_size=args.eval_batch_size, shuffle=False, 
                                 collate_fn=collate_ed_data)
        
        precision, recall, f1, results = evaluate(args, device, model, data_loader, save=args.save)

        log.info('[Best][{0:>10}] precision: {1:.2f}, recall: {2:.2f} , F1: {3:.2f}'.format(setname, precision, recall, f1))

        if setname != 'aida-train' and setname != 'aida-dev':
            scores.append(f1)

            if setname != 'aida-test':
                scores_oov.append(f1)

        if args.save:
            pred_dataset_path = os.path.join(args.save_path, "{}-kilt.jsonl".format(setname))

            with open(pred_dataset_path, "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

    avg = statistics.mean(scores)
    avg_oov = statistics.mean(scores_oov)

    log.info('[Best] Avg: {0:.2f}, Avg_OOD: {1:.2f}'.format(avg, avg_oov))


def build_second_step_examples(
        batch_examples, 
        batch_candidates, 
        batch_pred_ids, 
        batch_probs, 
        batch_num_candidates, 
        batch_gold_entities
    ):
    new_batch_examples = []

    for example, candidates, pred_id, probs, num_candidates, gold_entity \
        in zip(batch_examples, batch_candidates, batch_pred_ids, batch_probs, batch_num_candidates, batch_gold_entities):
        probs = probs[:num_candidates]

        paired_candidates = list(zip(candidates, probs))

        if len(candidates) == 0:
            pred_entity = "[NIL]"
        else:
            pred_entity = candidates[pred_id]

        example["paired_candidates"] = paired_candidates
        example["gold_entity"] = gold_entity
        example["pred_entity"] = pred_entity

        new_batch_examples.append(example)

    return new_batch_examples


def evaluate(args, device, model, data_loader, save=False):
    all_gold_entities = []
    all_candidates_list = []
    all_pred_ids = []
    all_num_candidates = []

    if save:
        all_examples = []

    total_batch = len(data_loader)
    model.eval()
    with tqdm(total=total_batch) as pbar:
        for step, batch in enumerate(data_loader, start=1):
            meta_data = batch['meta_data']

            gold_entities = meta_data['gold_entity']
            all_gold_entities += gold_entities
            
            candidates_list = meta_data['candidates']
            all_candidates_list += candidates_list

            del batch['gold_entity_positions']
            del batch['meta_data']

            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            with torch.no_grad():
                result = model(**batch)

            pred_ids, probs, num_candidates = result

            all_pred_ids += pred_ids
            all_num_candidates += num_candidates

            examples_list = meta_data['example_dict']

            if save:
                batch_examples = build_second_step_examples(
                    examples_list, 
                    candidates_list, 
                    pred_ids, probs, 
                    num_candidates, 
                    gold_entities
                )
                all_examples += batch_examples

            pbar.update(1)

    precision, recall, f1 = evaluate_ed_InKBF1(all_gold_entities, all_candidates_list, all_pred_ids, all_num_candidates)

    if not save:
        return precision, recall, f1, None

    return precision, recall, f1, all_examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    # Required Data
    parser.add_argument("--data_path", type=str, required=True, help="The input train corpus.")
    parser.add_argument("--save_path", type=str, required=True, help="")
    parser.add_argument('--log_file', default='run.log', help='')

    # General
    parser.add_argument('--seed', type=int, default=1013, help="")
    parser.add_argument('--epochs', type=int, default=5, help="")
    parser.add_argument('--eval', action='store_true', default=False, help='')
    parser.add_argument('--save', action='store_true', default=False, help='')
    parser.add_argument('--second_step', action='store_true', default=False, help='')
    parser.add_argument('--train_batch_size', type=int, default=4, help="")
    parser.add_argument('--eval_batch_size', type=int, default=8, help="")

    # Model
    parser.add_argument('--lm_name', type=str, default="allenai/longformer-large-4096",
                        choices=['allenai/longformer-base-4096', 'allenai/longformer-large-4096'], help='')
    parser.add_argument('--attention_window_size', type=int, default=256, choices=[16, 32, 64, 128, 256, 512], help='')
    parser.add_argument('--global_mode', type=int, default=4, choices=[1, 2, 3, 4], help='')
    parser.add_argument('--mean_resizing', type=lambda x: bool(strtobool(x)), default=True, help='')
    parser.add_argument('--model_type', type=str, default='dotproduct', 
                        choices=['dotproduct', 'classification', 'extractive'], help='')

    parser.add_argument('--scoring', type=str, default='sigmoid', choices=['softmax', 'sigmoid'], help='')
    
    parser.add_argument('--concat_mention', action='store_true', default=False, help='')
    parser.add_argument('--dropout_rate', help='Learning rate', default=0.1, type=float)
    
    parser.add_argument('--mask_mention', help='', default=False, action='store_true')
    parser.add_argument('--mask_mention_prob', help='Learning rate', default=0.0, type=float)

    parser.add_argument('--hidden_size', type=int, default=512, help='')

    parser.add_argument('--label_smooth', help='Learning rate', default=0.0, type=float)

    parser.add_argument('--checkpoint', type=str, default=None, help='')

    # Optimizer
    parser.add_argument('--optimizer', default='adamw', choices=["adamw", "radam"], help='adam adamw radam')
    parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--rlr', help='Remain(not contain LM) Learning rate', default=0.001, type=float)
    parser.add_argument('--beta1', help='Beta1 in ADAM', default=0.9, type=float)
    parser.add_argument('--beta2', help='Beta2 in ADAM', default=0.999, type=float)
    parser.add_argument('--eps', help='', default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--grad_clipping', help='Grad_clipping', default=10.0, type=float)
    parser.add_argument('--gradient_accumulation_steps', help='', default=1, type=float)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--lr_schedule", type=str, choices=["warmup_linear", "warmup_constant", "constant"],
                        default="warmup_linear", help="")
    parser.add_argument('--amp', help='', default=False, action='store_true')
    parser.add_argument('--max_patience', help='', default=3, type=float)
    
    parser.add_argument('--compile', type=lambda x: bool(strtobool(x)), default=False, help='')

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train(args)
