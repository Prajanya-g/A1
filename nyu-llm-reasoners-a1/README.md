# NYU Building LLM Reasoners Assignment 1: Basics

This assignment is adapted from Stanford CS336 ([original repository](https://github.com/stanford-cs336/)). All credit for its
development goes to the Stanford course staff. This README and all of the following code are adapted from theirs.

For a full description of the assignment, see the assignment handout at 
[a1.pdf](https://gregdurrett.github.io/courses/sp2026/a1.pdf)

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

### Run the TinyStories experiment

Experiment hyperparameters (vocab_size 10K, context_length 256, d_model 512, d_ff 1344, 4 layers / 16 heads, ~327M total tokens):

1. **Tokenize the data** (trains 10K BPE and writes `data/train_tokens.npy`, `data/valid_tokens.npy`):
   ```sh
   uv run python -m student.tinystories_tokenize_analysis
   ```

2. **Run training** (uses memmap data, checkpoints, and optional W&B logging):
   ```sh
   uv run python scripts/run_tinystories_experiment.py
   ```
   Add `--wandb` to log to Weights & Biases. Checkpoints are saved under `checkpoints/tinystories/`.

### Running on HPC (faster, GPU)

For cluster-specific details (partition names, time limits, modules), see **HPC_doc.pdf** (or your institution’s HPC docs).

1. **On the login node**: clone the repo, download data, install deps, then tokenize (once):
   ```sh
   cd /path/to/nyu-llm-reasoners-a1
   uv sync
   # download TinyStories into data/ (see above)
   uv run python -m student.tinystories_tokenize_analysis
   ```

2. **Option A – SLURM batch script**  
   Edit `scripts/run_tinystories_slurm.sh`: set `#SBATCH --partition=` to your GPU partition and adjust `--time` if needed. Then:
   ```sh
   mkdir -p logs
   sbatch scripts/run_tinystories_slurm.sh
   ```
   Logs go to `logs/tinystories_<jobid>.out` and `.err`.

3. **Option B – submitit (Python)**  
   Submits the same training job from Python:
   ```sh
   uv run python scripts/submit_tinystories_hpc.py --partition gpu
   ```
   Use `--partition` to match your cluster (e.g. `gpu`, `gpu_short`, `greene`). Optional: `--wandb`, `--time-min`, `--mem-gb`. Logs: `logs/submitit/`.

