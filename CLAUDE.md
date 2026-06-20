# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n RAG-Fuse python=3.10 -y
conda activate RAG-Fuse
pip install -r requirements.txt
python3 -m nltk.downloader punkt punkt_tab
```

`nmslib` must be installed from source (PyPI binary is broken):
```bash
pip install git+https://github.com/nmslib/nmslib/#subdirectory=python_bindings
```

## Running the Pipeline

Full pipeline for a dataset over a fold range:
```bash
bash run.sh <DATASET> <START_FOLD_IDX> <END_FOLD_IDX>
# e.g. bash run.sh ACM 0 4
```

Run individual tasks directly via Hydra overrides:
```bash
python main.py tasks=[sparse_retrieve] data=ACM data.folds=[0]
python main.py tasks=[fit] data=ACM model=RetrieverBERT data.folds=[0]
python main.py tasks=[predict] data=ACM model=RetrieverBERT data.folds=[0]
python main.py tasks=[eval] data=ACM model=RetrieverBERT data.folds=[0]
python main.py tasks=[fuse] data=ACM model=RetrieverBERT data.folds=[0]
python main.py tasks=[aggregate] data=ACM model=RetrieverBERT data.folds=[0]
python main.py tasks=[label_desc] data=ACM data.folds=[0]   # LLM stage, optional
python main.py tasks=[prompt_opt] data=ACM                  # LLM prompt tuning, optional
```

Debug a single batch:
```bash
python main.py tasks=[fit] data=ACM trainer.fast_dev_run=True
```

Monitor training:
```bash
tensorboard --logdir resource/log/ --port 6006
```

## Critical: Clean Resource Directories Between Experiments

The pipeline does **not** overwrite existing checkpoints or logs. Stale files cause training to resume from old checkpoints and produce incorrect results.

```bash
rm -rf resource/log/* resource/model_checkpoint/* resource/prediction/*
# or use the helper script:
bash resource/script/reset_resource.sh
```

## Architecture

RAG-Fuse frames multi-class text classification as a **ranking problem**. Documents and labels are encoded into a shared embedding space; labels are retrieved by similarity. The pipeline stages run sequentially:

1. **Sparse retrieval** (`SparseRetrieverHelper`) — BM25 via `retriv`, produces `.rnk` files in `resource/ranking/BM25_<DATASET>/`
2. *(Optional)* **Prompt optimisation** (`PromptOptimizerHelper`) — iteratively refines the LLM prompt template
3. *(Optional)* **Label description generation** (`LabelDescriptionHelper`) — calls AWS Bedrock (LLaMA) to produce enriched label text; outputs `labels_descriptions.pkl` per fold
4. **Dense retriever training** (`RetrieverFitHelper`) — fine-tunes BERT/RoBERTa with NTXent contrastive loss; checkpoint saved to `resource/model_checkpoint/`
5. **Prediction** (`RetrieverPredictHelper`) — generates text and label embeddings; written to `resource/prediction/`
6. **Evaluation** (`RetrieverEvalHelper`) — HNSW approximate nearest-neighbour search via `nmslib`; metrics written to `resource/result/`
7. **Fusion** (`RankingFusionHelper`) — combines BM25 and dense rankings with z-score normalisation + MNZ via `ranx`; outputs `resource/ranking/Fused_<MODEL>_<DATASET>/`
8. **Aggregation** (`RankingAggregationHelper`) — merges head/tail rankings and computes propensity-scored metrics (PSPrecision@k, PSnDCG@k)

### Key source modules

| Path | Role |
|------|------|
| `main.py` | Entry point; dispatches Hydra config to helper functions |
| `source/helper/Helper.py` | Base class with shared I/O (load samples, checkpoints, metrics) |
| `source/model/RetrieverModel.py` | PyTorch Lightning module: dual-encoder, NTXent loss, AdamW + linear warmup |
| `source/encoder/Retriever{BERT,RoBERTa}Encoder.py` | HuggingFace encoders with `ConcatenatePooling` (last 4 hidden layers → 3072-dim) |
| `source/loss/NTXentLoss.py` | In-batch contrastive loss, temperature 0.07 |
| `source/metric/RetrieverMetric.py` | MRR used as validation monitor for early stopping |
| `source/callback/RetrieverPredictionWriter.py` | Writes embeddings to disk during `predict` stage |

### Configuration (Hydra)

`setting/setting.yaml` is the root config; it composes `model` and `data` sub-configs from `setting/model/` and `setting/data/`. All CLI overrides follow Hydra dot-notation.

Key config knobs:
- `data.label_enhancement`: `RAW` | `PMI` | `LLM` — controls which label text is fed to the dense encoder
- `model.name`: used as a prefix for all output filenames; set to e.g. `LLM_RetrieverBERT` in the run scripts to distinguish experiments
- `fusion.model.target.name` / `fusion.model.source.name`: control which ranking files are fused

### Data format

All runtime data lives under `resource/dataset/<DATASET>/`:
- `samples.pkl` — list of `{idx, text_idx, text, labels, labels_ids}` dicts
- `relevance_map.pkl` — `{text_idx: [label_ids]}`
- `label_cls.pkl` / `text_cls.pkl` — `{idx: ["head"|"tail"]}`
- `propensities.pkl` — per-label propensity scores
- `fold_<N>/train.pkl`, `val.pkl`, `test.pkl` — sample ID splits
- `fold_<N>/labels_descriptions.pkl` — LLM-generated descriptions (optional, needed when `label_enhancement=LLM`)

Ranking files (`.rnk`) and result files (`.rts`) are pickled Python dicts and tab-separated CSVs respectively.
