# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repo contains experiments and artifacts for fine-tuning large language models to predict early autism risk from questionnaire data. The main work happens in Jupyter notebooks; the `autism_screening_mistral*` folders contain LoRA adapter checkpoints and configs for a fine-tuned Mistral-based model.

## Environment & dependencies

Development is notebook-driven and assumes a Python virtualenv (paths in the notebooks use `venv`).

### Python & core ML stack

From the notebooks:
- Python: CPython 3.13 (wheels like `cp313` are installed)
- Core libraries (GPU-enabled stack used in `llm_train.ipynb`):
  - `torch`, `torchvision`, `torchaudio` (CUDA 11.8 stack)
  - `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `trl`
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`

Typical setup (run in the project root inside an activated venv):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate bitsandbytes openpyxl pandas scikit-learn trl matplotlib seaborn
```

Additional dependencies used in `fine_tune_meta.ipynb`:

```bash
pip install kagglehub
```

These commands are all taken directly from `%pip install ...` cells in the notebooks.

## Common commands & workflows

There is no build system or standalone test runner; workflows are driven by notebooks.

### 1. Data acquisition & preprocessing

The Kaggle dataset is pulled in `fine_tune_meta.ipynb` via `kagglehub`:

```bash
pip install kagglehub
```

In the notebook, the dataset is downloaded with:

```python
import kagglehub
path = kagglehub.dataset_download("fabdelja/autism-screening-for-toddlers")
```

CSV files from that dataset (and a second toddler dataset) are then merged and converted into a chat-style JSONL training file `train.jsonl`.

### 2. Jupyter-driven experimentation

All core work (EDA, model training, evaluation, and tests) is inside two notebooks in the repo root:
- `llm_train.ipynb`
- `fine_tune_meta.ipynb`

Open these in your notebook environment (VS Code, Jupyter Lab, or similar) and execute cells top to bottom for the desired workflow (classical analysis vs. LLM fine-tuning).

### 3. Fine-tuning LLMs

There are two main fine-tuning flows represented:

1. **GPU Mistral fine-tuning (`llm_train.ipynb`)**
   - Base model: `mistralai/Mistral-7B-Instruct-v0.1` / `v0.2`.
   - Uses Hugging Face Transformers + PEFT LoRA + TRL `SFTTrainer`.
   - Produces LoRA checkpoints in `autism_screening_mistral/checkpoint-713` and `autism_screening_mistral/checkpoint-1426`, with training metrics stored in `trainer_state.json`.
   - A final, cleaned-up adapter is saved in `autism_screening_mistral_final/`.

2. **CPU Llama fine-tuning (`fine_tune_meta.ipynb`)**
   - Base model: `meta-llama/Llama-3.2-1B-Instruct`.
   - Converts the merged screening data to `train.jsonl` with a `messages` chat schema, then maps each example to a flat `text` field.
   - Uses TRL `SFTTrainer` with a LoRA `LoraConfig` targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`.
   - Saves a local fine-tuned adapter and tokenizer to `./llama-autism-cpu` (path referenced in the notebook; not committed here).

Running training is done by executing the relevant training cells in each notebook; there is no separate CLI entrypoint.

### 4. Evaluation & testing

Evaluation is also notebook-based:

- `llm_train.ipynb` builds a data-driven test suite around the fine-tuned Mistral model, using:
  - Real test data sampled from the held-out test split.
  - Synthetic but realistic edge-case prompts.
  - Helper functions like `create_prompt(...)` and `predict_autism_risk(...)` to generate predictions and compute accuracy.
- `fine_tune_meta.ipynb` includes:
  - Small synthetic evaluation datasets for the `./llama-autism-cpu` model, evaluated with `accuracy_score` and `classification_report` from scikit-learn.
  - A `unittest.TestCase` (`TestAutismScreening`) that verifies dataset creation, model loading, prediction generation, metric computation, and results saving.

There is currently **no standalone test command** (e.g., `pytest` or `python -m unittest` on a `.py` file). To re-run or narrow tests, open `fine_tune_meta.ipynb`, locate the `TestAutismScreening` cell, and execute/modify it directly.

## Repository structure & architecture

High-level structure (omitting generated model weights and obvious notebook details):

- **Root**
  - `README.md`: short one-line description of the project goal.
  - `llm_train.ipynb`: end-to-end workflow for:
    - Loading and cleaning two autism screening CSV datasets.
    - Standardizing schema (`A1`–`A10`, `Age`, `Sex`, `Jaundice`, `Family_ASD`, label column).
    - Exploratory analysis and classical ML-style evaluation of screening question patterns.
    - Fine-tuning a Mistral 7B Instruct model with LoRA using TRL `SFTTrainer`.
    - Building a fairly elaborate, data-driven evaluation harness on top of the fine-tuned model (real test cases, synthetic scenarios, and comprehensive metrics reporting).
  - `fine_tune_meta.ipynb`: more focused LLM fine-tune + evaluation pipeline for a smaller Llama model, including:
    - Data acquisition via Kaggle (`kagglehub`), merge of two toddler autism screening datasets, and JSONL chat-format generation (`train.jsonl`).
    - LoRA fine-tuning of `meta-llama/Llama-3.2-1B-Instruct` on CPU with TRL `SFTTrainer`.
    - Evaluation pipelines that construct synthetic chat-based test examples, generate predictions, and compute metrics with scikit-learn.
    - An embedded `unittest` suite (`TestAutismScreening`) that exercises dataset creation, model loading, prediction functions, metric calculation, and file output.

- **`autism_screening_mistral/`**
  - Contains intermediate LoRA checkpoints from Mistral fine-tuning (`checkpoint-713`, `checkpoint-1426`).
  - Each checkpoint includes:
    - `adapter_config.json`: PEFT LoRA configuration targeting `q_proj`, `k_proj`, `v_proj`, `o_proj` for a `CAUSAL_LM` task, with `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, and `base_model_name_or_path` set to `mistralai/Mistral-7B-Instruct-v0.1`.
    - Tokenizer metadata (`tokenizer*.json`) reusing a LLaMA-compatible tokenizer.
    - `trainer_state.json` capturing the full training log history (learning rate schedule, losses, eval metrics, and best checkpoint information).
    - A template Hugging Face model card `README.md` (mostly unfilled boilerplate as of now).

- **`autism_screening_mistral_final/`**
  - Mirrors the PEFT config structure of a Mistral LoRA adapter, but represents the final/best-performing adapter chosen from training.
  - Intended as the directory to load at inference time when attaching the autism-screening adapter to the base `mistralai/Mistral-7B-Instruct-v0.1` model.

## Using the fine-tuned adapters (conceptual)

Although there is no standalone script checked in for inference, the notebooks demonstrate and configure the following pattern:

- Base model: `mistralai/Mistral-7B-Instruct-v0.1`.
- PEFT LoRA adapters stored under `autism_screening_mistral/` and `autism_screening_mistral_final/`.
- Tokenizer configuration uses a LLaMA-compatible tokenizer (see `tokenizer_config.json` in `autism_screening_mistral_final/`).

To extend this repository, new scripts or notebooks should follow the existing convention:
- Use Hugging Face Transformers to load the base model.
- Use PEFT to load the LoRA adapter weights from `autism_screening_mistral_final/`.
- Construct prompts in the same format as the existing `train.jsonl`/evaluation examples ("Patient screening info:" followed by answers to A1–A10, demographic metadata, and the final question asking for YES/NO risk).

## Notes for future changes

- There is currently no `requirements.txt` or `pyproject.toml`; all dependency information lives in `%pip install ...` cells and import statements in the notebooks.
- Tests and evaluation harnesses are embedded in notebooks rather than in separate `.py` files. If you extract them into modules, consider adding a proper test runner (e.g., `pytest` or `python -m unittest`) and updating this file with the exact commands.
