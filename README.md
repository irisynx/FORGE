# FORGE — Fukui-Oriented Reaction Graph Editor

FORGE is a graph-edit neural model that predicts **which chemical bonds change, and by
how much**, when a reaction takes place. Given the reactant molecular graphs together
with the reaction context — environment molecules, a free-text experimental procedure,
temperature/time, and acid/base flags — FORGE outputs a set of bond-order edits that
transform reactants into products.

Rather than emitting product SMILES token by token, FORGE localizes the reaction to a
small set of atoms and edits the molecular graph directly. It (1) predicts
reaction-center atoms with a DETR-style set-prediction head, (2) constructs a sparse set
of candidate bonds from those atoms, and (3) lets the candidate bonds attend to one
another through an Edit Transformer that predicts a 7-class bond-order change
(`0, ±1, ±2, ±3`) for each. The graph encoder is a 6-layer GATv2 network **pretrained on
quantum-chemical electronic-structure targets** (condensed Fukui indices, partial
charges, HOMO/LUMO energies), which gives the model an inductive bias toward the frontier
reactivity that governs where bonds break and form.

This repository provides the complete code to reproduce FORGE: the **data pipeline**,
**encoder pretraining**, **model training**, and the **inference / evaluation** pipeline
(constrained decoding plus a product-aware re-ranker), on both the Open Reaction Database
(ORD) and the USPTO-480k benchmark.

---

## Table of contents

- [Repository structure](#repository-structure)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Data](#data)
  - [Data availability](#data-availability)
  - [Reaction data format](#reaction-data-format)
- [Reproduction workflow](#reproduction-workflow)
  - [Stage 1 — Preprocessing](#stage-1--preprocessing)
  - [Stage 2 — Encoder pretraining](#stage-2--encoder-pretraining)
  - [Stage 3 — Model training](#stage-3--model-training)
  - [Stage 4 — Inference and evaluation](#stage-4--inference-and-evaluation)
- [Model architecture](#model-architecture)
- [Configuration reference](#configuration-reference)
- [Results](#results)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Repository structure

```
FORGE/
├── README.md
├── requirements.txt
├── LICENSE
├── preprocessing/      Raw data sources  ->  PyG .pt chunks
│   ├── 00–04_*         Molecular electronic-structure data (DFT / xTB -> PyG Data)
│   ├── 09,20,21_*,23_* ORD reaction data (download / parse / clean / build)
│   └── 24–28_*         USPTO-480k benchmark preprocessing
├── pretraining/        GATv2 encoder pretraining on electronic-structure targets
│   ├── 10–12_*
│   └── forge_pretrain_utils.py   Training-curve / electronic-structure plots
├── training/           FORGE model training
│   ├── forge_rc_modules.py       Reaction-center set-prediction head + Hungarian loss
│   ├── forge_edit_modules.py     Bond-edit set-prediction head
│   ├── forge_role_signals.py     Reagent-role signal features (SMARTS/keyword based)
│   ├── train_forge.py            Training entry point (ORD)
│   └── train_forge_uspto.py      Training entry point (USPTO-480k)
└── evaluation/         Inference + evaluation
    ├── forge_decode_core.py      Constrained top-N decoding engine
    ├── forge_product_rerank.py   Product-plausibility features + re-ranker
    ├── evaluate_decoding.py      Exact-match evaluation (greedy / top-N + product hook)
    ├── dump_rerank_features.py   Dump per-candidate re-ranker features to feats.npz
    ├── train_eval_reranker.py    Train the re-ranker MLP + top-N evaluation
    └── eval_reranker_ensemble.py Multi-seed re-ranker ensemble (headline result)
```

Scripts are numbered in execution order; run them top to bottom within each stage.

---

## System requirements

**Operating system.** Developed and tested on Linux (Ubuntu). No OS-specific
dependencies; any platform supported by PyTorch should work.

**Hardware.**
- A CUDA-capable GPU is required for pretraining, training, and decoding. The released
  model was trained on a single **NVIDIA RTX 4090 (24 GB)**; ~16 GB is sufficient with
  the default `loader_batch_size = 4` and gradient accumulation.
- The electronic-structure preprocessing (`00–04`) is CPU-bound and depends on the
  external **xTB** semi-empirical quantum-chemistry engine.
- ~50 GB free disk is recommended for the full ORD `.pt` corpus and checkpoints.

**Software.** The released results were produced with the following environment; other
recent versions are expected to work. Exact pins are in `requirements.txt`.

| Package | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.4.0 (CUDA 12.4) |
| PyTorch Geometric | 2.7.0 |
| torch-scatter | matched to torch/CUDA |
| RDKit | 2025.09.3 |
| HuggingFace Transformers | 4.46.3 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| beautifulsoup4, spacy | preprocessing only |
| xTB (external) | preprocessing only |

A local copy of **SciBERT** (`allenai/scibert_scivocab_uncased`) encodes the experimental
procedure text; place it at `./scibert_local` (see [Installation](#installation)).

---

## Installation

Typical install time on a workstation with a warm package cache: **5–15 minutes**
(dominated by the PyTorch + PyG download).

```bash
# 1. Create an environment
conda create -n forge python=3.11 -y
conda activate forge

# 2. Install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install PyTorch Geometric + torch-scatter for the same torch/CUDA build
pip install torch_geometric==2.7.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# 4. Remaining dependencies
pip install -r requirements.txt

# 5. (Preprocessing only) install xTB
conda install -c conda-forge xtb

# 6. Download SciBERT once and point the config at it
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('allenai/scibert_scivocab_uncased', local_dir='./scibert_local')"
```

---

## Data

### Data availability

FORGE is trained on public reaction datasets; no proprietary data is required.

| Dataset | Source | Used for |
| --- | --- | --- |
| Open Reaction Database (ORD) | https://open-reaction-database.org | Primary training / evaluation |
| USPTO-480k (Jin split) | Public reaction-prediction benchmark | Cross-domain benchmark |

The processed PyG `.pt` datasets and the model checkpoints are archived on Zenodo:

> **Zenodo DOI:** [`10.5281/zenodo.21125167`](https://doi.org/10.5281/zenodo.21125167) *(record released with the paper)*

The archive contains:

| File | Contents | Extract to |
| --- | --- | --- |
| `ord_train.tar.zst` | ORD training set (PyG `.pt` chunks) | `./ord_data/ord_train/` |
| `ord_heldout.tar.zst` | ORD held-out val/test set | `./ord_data/ord_heldout/` |
| `uspto_480k.tar.zst` | USPTO-480k (train/valid/test) | `./uspto_480k/` |
| `forge_pretrained_encoder.pth` | Pretrained GATv2 encoder | `./checkpoints/` |
| `forge_ord_model.pth` | Trained ORD model | `./results_editgnn_v34_E1/` |
| `forge_uspto_model.pth` | Trained USPTO model | — |

Decompress a dataset with: `zstd -dc ord_train.tar.zst | tar -xf -`. The default `CONFIG`
paths in the training scripts match the extraction locations above.

Alternatively, the `preprocessing/` scripts regenerate the processed corpus from the public raw sources:
`09_download_ord.py` fetches the ORD protobuf datasets, and `24_uspto_jin_to_pt.py`
ingests the USPTO Jin split. The electronic-structure preprocessing (`00–04`) additionally
consumes per-molecule quantum-chemical property files (DFT / xTB output) to build the
encoder pretraining targets.

### Reaction data format

Each reaction `.pt` file holds a list of PyG `Data` objects (≈64 reactions per chunk):

| Field | Shape / type | Description |
| --- | --- | --- |
| `x` | `[N, 8]` int | 8 discrete atom features: AtomicNum, Degree, FormalCharge, Hybridization, Aromatic, Mass, Chirality, TotalNumHs |
| `edge_index` | `[2, E]` long | Molecular graph connectivity |
| `edge_attr` | `[E]` int | Bond type (1=single, 2=double, 3=triple, 4=aromatic) |
| `y_delta` | `[N, N]` long | Ground-truth bond-order change matrix, encoded as 7 classes (`0, +1, −1, +2, −2, +3, −3`) |
| `env_x` | `[M, 1024]` | Morgan fingerprints of environment (reagent/solvent/catalyst) molecules |
| `proc_text` / `work_text` | str | Experimental procedure / work-up text (encoded by SciBERT) |
| `temp_id`, `time_val`, `ab_flags` | scalar / small vector | Temperature bin, time value, acid/base flags |

Atom-feature vocabulary sizes are fixed to
`ATOM_FEATURE_DIMS = [120, 15, 15, 10, 10, 100, 10, 10]` to match the pretrained encoder
and must not be changed without re-pretraining.

---

## Reproduction workflow

The pipeline has four stages; each stage's scripts are numbered and run in order. For a
**quick end-to-end smoke test**, set `data_ratio` to a small fraction (e.g. `0.05`) in the
training `CONFIG` before Stage 3 — this trains on ~5% of the data and finishes in minutes.

### Stage 1 — Preprocessing

Molecular electronic-structure data (per-molecule `.pt` used as pretraining targets):

```bash
python preprocessing/00_preprocess_dft.py
python preprocessing/01_orbital_features.py
python preprocessing/02_data_update.py
python preprocessing/03_atom_features_update.py
python preprocessing/04_unified_processing.py
```

Reaction data (ORD → cleaned CSV → PyG `.pt` chunks):

```bash
python preprocessing/09_download_ord.py            # download ORD protobuf
python preprocessing/20_parse_ord.py               # protobuf -> CSV
python preprocessing/21_0_text_clean.py
python preprocessing/21_1_dynamic_clean_acid_base.py
python preprocessing/21_2_conservation_stats.py
python preprocessing/21_3_extract_text_verbs.py
python preprocessing/21_4_final_mapping_filter.py
python preprocessing/21_5_low_yield_clean.py
python preprocessing/21_6_unchanged_product_clean.py
python preprocessing/21_7_mapping_check.py
python preprocessing/21_8_change_count_classify.py
python preprocessing/23_build_reaction_pt_chunks.py   # emit final .pt chunks
```

USPTO-480k benchmark (optional, for the cross-domain experiments):

```bash
python preprocessing/24_uspto_jin_to_pt.py
python preprocessing/25_uspto_rxnmapper_remap.py
python preprocessing/26_uspto_remap_audit.py
python preprocessing/27_uspto_remap_threshold_stats.py
python preprocessing/28_uspto_chem_check.py
```

**Output.** PyG `.pt` chunk files under a processed-data directory (e.g.
`./ord_data/ord_train/`, `./ord_data/ord_heldout/`) in the [format above](#reaction-data-format).

### Stage 2 — Encoder pretraining

```bash
python pretraining/10_pretrain_encoder.py    # Fukui / charge / HOMO / LUMO regression
python pretraining/11_pretrain_atom_aux.py   # auxiliary atom-feature task
python pretraining/12_pretrain_global_aux.py # auxiliary global-information task
```

**Output.** The pretrained encoder checkpoint `model_epoch_200.pth`. Place it at
`./checkpoints/model_epoch_200.pth` (or update `elec_ckpt_path` in the training config).

### Stage 3 — Model training

```bash
# Primary model on ORD
python training/train_forge.py

# Cross-domain benchmark on USPTO-480k
python training/train_forge_uspto.py
```

Training runs in two phases (reaction-center head first, then joint RC + edit training)
for 30 epochs by default. On a single RTX 4090, full-corpus ORD training takes on the
order of a few days; the 5% smoke run completes in minutes.

**Output.** Under `save_dir` (default `./results_editgnn_v34_E1/`): `history.csv`
(per-epoch metrics), `model_best.pth` (best checkpoint), and a snapshot of the training
script for provenance.

### Stage 4 — Inference and evaluation

Reported accuracy is **exact-match top-1** (the predicted set of bond edits reproduces the
reference product). The evaluation scripts import the trained model dynamically via
`--module train_forge`.

**(a) Constrained decoding — base model accuracy.**

```bash
python evaluation/evaluate_decoding.py \
    --module train_forge \
    --ckpt ./results_editgnn_v34_E1/model_best.pth \
    --topk 3 \
    --enable_product_hook
```

`--topk N` applies the top-N predicted edits (N=3 is the reported operating point). Use
`--test_set <path>` or `--val_split` to select the evaluation split (see `--help`).

**(b) Product-aware re-ranker — headline accuracy.** Dump per-candidate features, then
train and evaluate the re-ranker MLP; the ensemble script averages several seeds.

```bash
# 1. Dump re-ranker features (GT-free) to feats.npz
python evaluation/dump_rerank_features.py \
    --module train_forge \
    --ckpt ./results_editgnn_v34_E1/model_best.pth \
    --out ./rerank/feats.npz

# 2. Train the re-ranker MLP and evaluate top-N exact-match
python evaluation/train_eval_reranker.py \
    --feats ./rerank/feats.npz \
    --out ./rerank/topk_table.json \
    --topk 1,3,5,10

# 3. Multi-seed ensemble (headline number)
python evaluation/eval_reranker_ensemble.py \
    --feats ./rerank/feats.npz \
    --out ./rerank/ensemble.json
```

---

## Model architecture

| Component | Specification |
| --- | --- |
| Graph encoder | 6-layer GATv2, hidden dim 512, 8 attention heads, edge-embedding dim 64, no JK-Net (frozen after pretraining, with a trainable adapter) |
| Reaction-center head | DETR-style set-prediction decoder + Hungarian matching (`forge_rc_modules.py`) |
| Candidate-bond edit head | DETR-style edit decoder over candidate bonds, 8 heads (`forge_edit_modules.py`) |
| Edit Transformer | 4 layers of candidate-bond self-attention |
| Text encoder | SciBERT (procedure / work-up text) |
| Environment encoder | Morgan fingerprints (`[M, 1024]`) |
| Reagent-role signals | SMARTS/keyword reactivity features (`forge_role_signals.py`) |
| Output | Per-bond 7-class bond-order change (`0, ±1, ±2, ±3`) |

The encoder structure is fixed to match the released pretrained checkpoint. Changing the
number of layers, hidden dim, heads, or `ATOM_FEATURE_DIMS` requires re-running Stage 2.

---

## Configuration reference

All hyperparameters live in the `CONFIG` dictionary at the top of each training script.
The most commonly adjusted keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `data_dir` | processed ORD directory | Location of training `.pt` chunks |
| `elec_ckpt_path` | `./checkpoints/model_epoch_200.pth` | Pretrained encoder checkpoint |
| `bert_model` | `./scibert_local` | Local SciBERT path |
| `save_dir` | `./results_editgnn_v34_E1` | Output directory (history, checkpoints) |
| `data_ratio` | `1.0` | Fraction of data to use (set small for quick runs) |
| `loader_batch_size` | `4` | Per-step batch size |
| `accum_steps` | `4` | Gradient-accumulation steps (effective batch = `batch × accum`) |
| `num_workers` | `6` | DataLoader worker processes |
| `epochs` | `30` | Total training epochs |
| `phase1_start_epoch` / `phase2_start_epoch` | `6` / `1` | Two-phase training schedule |
| `use_role_signals` | `True` | Enable reagent-role reactivity features |
| `resume_from_epoch` | — | Resume from a checkpoint epoch |

---

## Results

Top-1 exact-match accuracy on held-out test sets:

| Benchmark | Test reactions | Base (constrained decoding) | + Product-aware re-ranker |
| --- | --- | --- | --- |
| ORD (held-out) | 64,484 | ~0.906 | **~0.951** |
| USPTO-480k (Jin test) | 39,768 | ~0.915 | **~0.953** |

The base column corresponds to `evaluate_decoding.py`; the re-ranked column corresponds
to the Stage 4(b) ensemble. See the paper for the complete evaluation protocol, ablations,
and interpretability analyses.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

---

## Citation

If you use FORGE or this code, please cite:

```bibtex
@misc{forge2026,
  title        = {FORGE: A Fukui-Oriented Reaction Graph Editor for interpretable
                  reaction-outcome prediction},
  author       = {Gong, Zheng and Zhang, Baicheng and Jiang, Jun and Luo, Yi and Zhang, Guoqing},
  year         = {2026},
  howpublished = {Preprint},
  doi          = {<preprint DOI — to be added>}
}
```

*Currently a preprint; the journal reference and DOI will be updated upon acceptance.*

---

## Contact

For questions about the code, or to request the processed datasets and pretrained
checkpoints ahead of the Zenodo release, please open an issue on this repository or
contact the corresponding author (see the paper).
