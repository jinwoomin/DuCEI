# DuCEI

This repository contains the source code for **DuCEI**, provided as supplementary material for code inspection during review. The package includes the main model, preprocessing logic, data loading utilities, optimization utilities, and shell scripts for the main experimental workflow.

This supplementary package is not intended to be a fully self-contained release. Large external resources, including the Wikipedia dump database, processed entity information, intermediate data files, and trained checkpoints, are not included in this submission. These resources will be made publicly available in a later release.

## Environment

The code was prepared with Python 3.11.14. We recommend creating a dedicated conda environment before installing dependencies:

```bash
conda create -n DuCEI python=3.11.14
conda activate DuCEI
pip install -r requirements.txt
```

Python 3.11 can also be used if Python 3.11.14 is not available from your conda channels.

PyTorch installation can depend on the local CUDA version. If needed, install the appropriate PyTorch build for your GPU environment before installing the remaining dependencies.

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

The code expects external data and Wikipedia resources that are not included in this review package:

- KILT-style entity disambiguation datasets
- Wikipedia2Vec dump database, e.g., `enwiki-latest.db`
- Processed Wikipedia entity information, e.g., `wiki_title_info.json`
- Stage-1 prediction files used for second-stage preprocessing
- Trained model checkpoints

The KILT-style dataset files can be obtained using the dataset download script provided by GENRE:

```text
https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh
```

Please refer to the GENRE repository for dataset download instructions and licensing terms.

The remaining resources, including the Wikipedia dump database, processed Wikipedia entity information, intermediate files, and trained checkpoints, are omitted from the supplementary material due to size and release constraints. We plan to release these resources with the camera-ready version.

## Main Workflow

The shell scripts under `scripts/` show the commands and hyperparameter settings used for the main workflow. They assume they are launched from the `DuCEI/` directory and that the required external resources have been placed at the paths referenced in each script.

First, preprocess the stage-1 input data:

```bash
bash scripts/preprocess/stage1/preprocess.sh
```

Then train the stage-1 model. Set `CUDA_VISIBLE_DEVICES` according to the local GPU environment:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage1/train.sh
```

After training, run stage-1 evaluation with prediction saving enabled. The saved prediction files are used as input for second-stage preprocessing:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage1/eval.sh
```

Next, preprocess the second-stage data from the stage-1 prediction files:

```bash
bash scripts/preprocess/stage2/preprocess.sh
```

Then train the second-stage model. This stage is initialized from a stage-1 checkpoint, so set the `checkpoint` variable in `scripts/stage2/train.sh` before running:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage2/train.sh
```

Finally, evaluate the second-stage model and save predictions:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stage2/eval.sh
```

The scripts are included primarily to document the experimental commands used in the paper. Reviewers can inspect or run them after preparing the required external resources.

## Notes

- `main.py` contains the training and evaluation loop.
- `preprocess.py` converts KILT-style examples and stage-1 predictions into model instances.
- `model.py` contains the model variants used in the experiments.
- `utils/evauation_utils.py` keeps the original project filename for compatibility with existing imports.
