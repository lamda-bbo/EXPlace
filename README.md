# EXPlace

**Expertise Can Be Helpful for Reinforcement Learning-based Macro Placement**

*Published as a conference paper at* ***ICLR 2026***

Chengrui Gao, Yunqi Shi, Ke Xue, Ruo-Tong Chen, Siyuan Xu, Mingxuan Yuan, Chao Qian†, Zhi-Hua Zhou

Nanjing University | Huawei Noah's Ark Lab

---

## 🌟 Highlights

- 🏆 **State-of-the-art PPA**: Best average rank across all critical metrics (rWL, WNS, TNS, DRC) on both ICCAD 2015 and OpenROAD benchmarks
- ⚡ **7.74% TNS improvement** and **32.53% WNS improvement** over the runner-up on ICCAD 2015
- 🔄 Outperforms advanced analytical, black-box optimization, and prior RL-based placers

---

## 📖 About the Paper

Existing RL-based macro placement methods optimize oversimplified proxy objectives (e.g., macro HPWL) and neglect expert knowledge accumulated through years of engineering practice, resulting in placements that deviate significantly from expert-designed solutions. EXPlace is the first paper to bridge this gap and advance the performance of RL placers.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## Installation

1. **Clone the repository:**
  ```bash
   git clone <repository_url>
   cd explace
  ```
2. **Download benchmarks, DREAMPlace folder and preprocessed data** into the project root: https://drive.google.com/drive/folders/110ZVI-eR-HlzOflRsHC_XgpEJXT0uELN?usp=share_link
3. **Load the Docker image** (`.tar.gz`):
  ```bash
   docker load -i explace_env.tar.gz
   bash script/start_docker.sh
  ```
   Then run your commands inside a container from that image.
4. **Make DREAMPlace** for global placement:
  ```bash
  bash script/make.sh
  ```
---

## Quick Start

**Train** on a single design (e.g. `superblue3`) with the provided config:

```bash
python -m src.main --benchmark superblue3 --config iccad --gpu 0 --seed 3
```

**Evaluate** a trained model:

```bash
python -m src.main --benchmark superblue3 --config iccad --test \
  --model_path rl_logs/superblue3/<timestamp>/model/best_tns_model.pt
```

---

## Training

Run training from the **project root**:

```bash
python -m src.main --benchmark <design_name> --config <config_name> [options]
```


| Argument      | Required | Description                                                      |
| ------------- | -------- | ---------------------------------------------------------------- |
| `--benchmark` | Yes      | Design name (e.g. `superblue3`, `superblue1`)                    |
| `--config`    | Yes      | Config name without `.yaml`; file is `config/<config_name>.yaml` |
| `--gpu`       | No       | GPU id (overrides `gpu` in config)                               |
| `--seed`      | No       | Random seed (overrides `seed` in config)                         |
| `--debug`     | No       | Debug mode: single env, drops into pdb                           |


**Example:**

```bash
python -m src.main --benchmark superblue3 --config iccad --gpu 0 --seed 3
```

**Batch training** (multiple designs / GPUs) is demonstrated in `script/run_RL.sh`:

```bash
bash script/run_RL.sh
```

Outputs are written under `log_dir` (set in config; default `./`) as:

- `rl_logs/<benchmark>/<timestamp>/`
  - `model/`: e.g. `current_model.pt`, `best_tns_model.pt`, `best_gp_hpwl_model.pt`
  - `placement/`: best placement `.def` files
  - `visualization/`: placement figures
  - TensorBoard logs, `runtime.csv`, `eval_metrics.csv`, `best_metrics.csv`

---

## Test

To run **inference** with a trained model:

```bash
python -m src.main --benchmark <design_name> --config <config_name> --test --model_path <path_to_.pt> [--visualize]
```


| Argument       | Required | Description                                                              |
| -------------- | -------- | ------------------------------------------------------------------------ |
| `--benchmark`  | Yes      | Same design name as used in training                                     |
| `--config`     | Yes      | Same config name as in training (for env and evaluation setup)           |
| `--test`       | Yes      | Enables evaluation mode                                                  |
| `--model_path` | Yes      | Path to checkpoint (e.g. `best_tns_model.pt` or `best_gp_hpwl_model.pt`) |


Evaluation produces `test_dmp.def`, `test_dmp.png`, and timing metrics (GP-HPWL, post-placement TNS, post-placement WNS) under the run’s log directory. Note that evaluating post-route performance **requires additional commercial or academic tools for routing and sign-off**.

---

## Configuration

The active config is selected by `--config <name>` (without `.yaml`); the file is `config/<name>.yaml`. Command-line flags override config for `--gpu` and `--seed`.

### `config/iccad.yaml`

This config is used for ICCAD / SuperBlue-style benchmarks. Main fields:


| Section        | Key                                           | Description                                                                                |
| -------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Training**   | `episode`                                     | Total training episodes (e.g. 1500).                                                       |
|                | `rollout_batch_size`                          | Number of parallel Ray environments (e.g. 10).                                             |
|                | `batch_size`                                  | PPO update batch size (e.g. 256).                                                          |
|                | `buffer_size`                                 | Trajectory buffer size.                                                                    |
| **PPO**        | `clip_param`, `max_grad_norm`, `ppo_epoch`    | PPO hyperparameters.                                                                       |
|                | `gamma`, `lr`                                 | Discount factor and learning rate.                                                         |
| **Reward**     | `used_masks`                                  | Reward terms, e.g. `["reg", "hier", "df", "wire", "port", "pos"]`.                         |
|                | `trade_off_coeff`                             | Weights for each term (same length as `used_masks`), e.g. `[0.45, 0.2, 0.15, 0.15, 0.05]`. |
|                | `dataflow_cutoff`, `halo`, `soft_coefficient` | Dataflow cutoff, macro halo, soft constraint.                                              |
| **Evaluation** | `use_dmp_for_evaluation`                      | Use DREAMPlace for periodic GP-HPWL evaluation.                                            |
|                | `use_timer_for_evaluation`                    | Use OpenTimer for timing.                                                                  |


To customize, copy `config/iccad.yaml` to a new file and run with `--config <new_name>`.

---

## Project Structure

```
explace/
├── config/                 # YAML configs (e.g. iccad.yaml)
├── src/
│   ├── main.py             # Entry: training and evaluation
│   ├── agent.py            # PPO agent
│   ├── env_utils.py        # Env creation, Ray, preprocessing
│   ├── problem_instance.py # Problem instance, DREAMPlace interface
│   ├── state_parsing.py    # State parsing
│   ├── place_env/          # Placement environment
│   └── model/              # Actor / critic networks
├── script/
│   ├── make.sh             # Build DREAMPlace (cmake, make, install)
│   ├── run_RL.sh           # Batch training: multiple designs / GPUs
│   └── start_docker.sh     # Start Docker with GPU and mount repo to /workspace
├── dataflow_graph/         # Dataflow graphs (from Google Drive)
├── dataflow_info/         # Precomputed dataflow (from Google Drive)
├── processed_data/        # Cached env params (from Google Drive)
├── benchmarks/            # Benchmarks (from Google Drive)
└── DREAMPlace/            # Global placer and post-placement timer (from Google Drive)
```

Log and checkpoint paths follow `log_dir` and the run timestamp, typically `rl_logs/<benchmark>/<timestamp>/`.

### Citation

```bibtex
@inproceedings{gao2026expertise,
  title={Expertise Can Be Helpful for Reinforcement Learning-based Macro Placement},
  author={Gao, Chengrui and Shi, Yunqi and Xue, Ke and Chen, Ruo-Tong and Xu, Siyuan and Yuan, Mingxuan and Qian, Chao and Zhou, Zhi-Hua},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

