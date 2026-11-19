# FlexSLM for medical question answering

This repository is based on [sudaksh14/FlexSLM](https://github.com/sudaksh14/FlexSLM). The original FlexSLM code and its full history are in that repository.

## Added in this repository

- Qwen3-1.7B as backbone, and export of every level as a Hugging Face model (`export_flex_hf.py`)
- Fine-tuning with knowledge distillation and FaithMed's schedule, with 3 or 10 levels (`sft.py`)
- `kv_group` option: keeps the query / key-value head pairing of a pretrained GQA model on every level
- Importance reordering of Qwen3 (`reorder_qwen3.py`)
- Medical evaluation: cloze tasks (`medcloze/`) and FaithMed's seven exams (`eval_gen.py`)
- Paired McNemar test between two models (`compare_paired.py`)
- Pretraining data: RefinedWeb, GneissWeb and PubMed Central

## Data

- RefinedWeb: streamed from Hugging Face (`tiiuae/falcon-refinedweb`) during training.
- GneissWeb: `sbatch download_bloom_job.sh` downloads IBM's bloom filter (27 GB). Training streams FineWeb and keeps the matching documents.
- PubMed Central: `python fetch_pmc_commercial_ids.py` writes the article list. Training downloads the articles and splits them.
- MedQA + MedMCQA (160M models): `python prepare_medqa_sft.py`
- FaithMed data (Qwen3): `python prepare_medmix_sft.py`. For 60,000 MedMCQA questions add `--out data/medmix_sft_mcqa60k --cap openlifescienceai/medmcqa=60000`.
- Reordered Qwen3 weights: `sbatch reorder.sh` (add `--heads` to reorder the attention heads too).

Configs are listed in `config/experiments.py` and `sft.py`.
