# DuCEI

This repository contains the source code for **DuCEI: When to Introduce Metadata in Cross-Entity Interaction for Entity Disambiguation**.

The repository includes the main model implementation, preprocessing logic, data loading utilities, optimization utilities, and shell scripts for the main experimental workflow.

Large external resources, including Wikipedia resources, processed entity information, intermediate prediction files, evaluation datasets, and trained checkpoints, are not distributed in this repository.

## Environment

The code was prepared with Python 3.11.14. We recommend creating a dedicated conda environment before installing dependencies:

```bash
conda create -n DuCEI python=3.11.14
conda activate DuCEI
pip install -r requirements.txt
```

Python 3.11 can also be used if Python 3.11.14 is not available from your conda channels.

PyTorch installation can depend on the local CUDA version. If needed, install the appropriate PyTorch build for your GPU environment before installing the remaining dependencies.

The experiments were conducted with mixed-precision (`fp16`) training. Minor numerical differences may occur across GPU architectures, CUDA/PyTorch versions, and mixed-precision implementations.

## Repository Structure

```text
DuCEI/
├── main.py                         # Training and evaluation entry point
├── preprocess.py                   # Preprocessing entry point
├── model.py                        # DuCEI model definitions
├── requirements.txt                # Minimal runtime dependencies
├── data/                           # Dataset readers and tensorization utilities
├── utils/                          # Evaluation, logging, and optimization utilities
└── scripts/
    ├── preprocess/
    │   ├── stage1/preprocess.sh
    │   └── stage2/preprocess.sh
    ├── stage1/
    │   ├── train.sh
    │   └── eval.sh
    └── stage2/
        ├── train.sh
        └── eval.sh
```

## External Resources

The code expects external data and Wikipedia resources that are not included in this repository:

* KILT-style entity disambiguation datasets
* Wikipedia2Vec dump database, e.g., `enwiki-latest.db`
* Processed Wikipedia entity information, e.g., `wiki_title_info.json`
* Stage-1 prediction files used for second-stage preprocessing
* Trained model checkpoints

The KILT-style dataset files can be obtained using the dataset download script provided by GENRE:

https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh

Please refer to the GENRE repository and the original dataset distributions for download instructions and licensing terms. Evaluation datasets are not redistributed in this repository.

The Wikipedia dump database and processed entity information should be prepared locally and placed at the paths specified in the preprocessing scripts.

Stage-1 prediction files are generated locally by running Stage-1 evaluation and are subsequently used for Stage-2 preprocessing. Pre-generated intermediate prediction files and trained checkpoints are not included.

## Main Workflow

The shell scripts under `scripts/` provide the commands and default settings for the main experimental workflow. They assume that they are launched from the `DuCEI/` directory and that the required external resources have been placed at the paths referenced in each script.

The shell scripts provide default runnable settings. Hyperparameters such as mention masking probability can be adjusted to match specific experimental configurations reported in the paper.

First, preprocess the Stage-1 input data:

```bash
bash scripts/preprocess/stage1/preprocess.sh
```

Then train the Stage-1 model. Set `CUDA_VISIBLE_DEVICES` according to the local GPU environment:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage1/train.sh
```

After training, run Stage-1 evaluation with prediction saving enabled. The resulting prediction files are used as input for Stage-2 preprocessing:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage1/eval.sh
```

Next, preprocess the Stage-2 data using the Stage-1 prediction files:

```bash
bash scripts/preprocess/stage2/preprocess.sh
```

Then train the Stage-2 model. Stage 2 is initialized from a Stage-1 checkpoint generated during Stage-1 training. Set the `checkpoint` variable in `scripts/stage2/train.sh` accordingly before running:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage2/train.sh
```

Finally, evaluate the Stage-2 model and save predictions:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage2/eval.sh
```

## Notes

* `main.py` contains the training and evaluation loop.
* `preprocess.py` converts KILT-style examples and Stage-1 predictions into model instances.
* `model.py` contains the model variants used in the experiments.
* `utils/evauation_utils.py` keeps the original project filename for compatibility with existing imports.
* Exact numerical results may vary slightly depending on hardware, CUDA/PyTorch versions, and mixed-precision computation.
