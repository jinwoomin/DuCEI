from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import List
from torch.optim import SGD, Adam, AdamW, RAdam
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import get_constant_schedule

OPTIMIZER_CLASSES = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'radam': RAdam,
}

def get_decay_parameter_names(model) -> List[str]:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
    return decay_parameters

def create_optimizer(args, model):
    decay_parameters = get_decay_parameter_names(model)

    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_class = OPTIMIZER_CLASSES[args.optimizer]

    if args.optimizer in ['sgd', 'adam', 'adamw']:
        return optimizer_class(grouped_parameters, lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2), fused=True)
    else:
        return optimizer_class(grouped_parameters, lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))

def create_scheduler(args, optimizer, num_train_steps):
    warmup_steps = int(num_train_steps * args.warmup_proportion)

    if args.lr_schedule == "warmup_linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
    if args.lr_schedule == "warmup_constant":
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)
    if args.lr_schedule == "constant":
        return get_constant_schedule(optimizer)
    raise RuntimeError("Unsupported scheduler: " + args.lr_schedule)
