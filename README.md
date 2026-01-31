AlphaZero-lite Chess (PyTorch)
==============================

Overview
--------
This project trains a lightweight AlphaZero-style agent for chess using:
- a policy+value network (PyTorch),
- batched MCTS,
- self-play data generation,
- replay buffer training,
- simple evaluation vs a baseline GreedyBot.

Quickstart
----------
1) Create and activate a virtual environment.
2) Install dependencies:
   `pip install -r requirements.txt`
3) (Recommended) Install NumPy to avoid warnings and enable some ops:
   `pip install numpy`
4) Run training:
   `python azlite_chess_train.py`

Windows (RTX) Setup
-------------------
1) Install Python 3.12 x64.
2) Create and activate venv:
   `py -3.12 -m venv .venv`
   `.\\.venv\\Scripts\\activate`
3) Install CUDA build of PyTorch (pick the CUDA version shown on the official PyTorch
   "Start Locally" page for Windows/Pip/Python/CUDA):
   `pip uninstall -y torch torchvision torchaudio`
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12x`
4) Verify GPU is detected:
   `python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`
5) Run training:
   `python azlite_chess_train.py`

Checkpoints / Resume
--------------------
Training auto-saves to `out_azlite/`:
- `latest.pt` (most recent)
- `ckpt_iter_<N>.pt`

On startup, the script automatically resumes from `latest.pt` if it exists.
This includes model, optimizer, replay buffer, RNG state, and iteration index.

Configuration
-------------
All hyperparameters live in the `Config` dataclass near the top of
`azlite_chess_train.py`. Key groups:
- MCTS: `mcts_sims`, `cpuct`, `dirichlet_alpha`, `dirichlet_eps`, `mcts_eval_batch`
- Self-play: `games_per_iter`, `max_game_plies`, `temperature_moves`
- Training: `replay_max`, `train_steps_per_iter`, `batch_size`, `lr`, `weight_decay`
- Model: `channels`, `res_blocks`
- Eval: `eval_games`, `eval_every_iters`
- Logging: `log_train_every`

Logs
----
Each iteration logs:
- self-play sample counts and aggregate self-play stats
- training loss (policy/value), gradient norm, entropy/top-1 for target/predicted policy,
  and value mean/std vs targets
- eval vs GreedyBot (W/D/L, score, estimated Elo)
- timing breakdown (self-play/train/eval/save/total)

Troubleshooting
---------------
NumPy warning:
- If you see "Failed to initialize NumPy", install NumPy:
  `pip install numpy`

MPS (Apple GPU) notes:
- Dirichlet sampling is not implemented on MPS, so the code falls back to CPU
  for that step only.
- If other MPS ops fail, you can set:
  `PYTORCH_ENABLE_MPS_FALLBACK=1`
  This uses CPU fallback for unsupported ops (slower).

Stopping
--------
Stop with Ctrl+C. The latest checkpoint is saved every `save_every_iters`
iterations, so you can restart without losing progress.
