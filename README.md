# MALoRA — Mixture of Asymmetric LoRA for Code Generation

Fine-tuning Qwen2.5-Coder-3B-Instruct with a custom MoE-LoRA architecture combining sparse expert routing with asymmetric attention LoRA adapters.

---

## Architecture

- **Base model:** Qwen2.5-Coder-3B-Instruct (frozen)
- **MoE LoRA:** 8 experts per MLP layer, rank 8, top-2 sparse routing (DispatchMoERouter)
- **Attention LoRA:** rank 32 adapters on Q/K/V/O projections (optional, toggle in gemini.py)
- **Trainable params:** ~91M (MoE only) or ~106M (MoE + attention LoRA)
- **Total params:** ~3.5B

---

## Hardware Requirements

- **Recommended:** H100 (80GB) — batch size 32, ~3.5hrs for 100K/1ep
- **Minimum:** L40S (48GB) — batch size 12, ~10hrs for 100K/1ep
- **Platform:** Lightning AI Studio (SSH via VS Code)

---

## Setup

### 1. Clone the repo

```bash
cd /teamspace/studios/this_studio
git init
git remote add origin https://github.com/godofwar1007/MOELoRA
git pull origin main
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If anything is missing:
```bash
pip install bitsandbytes python-dotenv huggingface_hub wandb evalplus --break-system-packages
```

### 3. Login to WandB

```bash
wandb login
# paste your API key from wandb.ai/settings
```

Then in `gemini.py`, update this line to your WandB entity:
```python
os.environ["WANDB_ENTITY"] = "your-wandb-username"
```

### 4. Login to HuggingFace

```bash
huggingface-cli login
# paste your HF token from huggingface.co/settings/tokens
```

### 5. Verify GPU

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# should show 0 MiB used
```

---

## Configuration

### Dataset size — `config.py`

```python
TOTAL_SAMPLES = 30000   # change this: 10000 / 20000 / 30000 / 50000 / 100000
```

### Training hyperparameters — `training_config.py`

Key settings to adjust based on GPU:

```python
NUM_EPOCHS: int      = 3      # epochs — this matters more than data size
TRAIN_BATCH: int     = 32     # H100: 32, L40S: 12
EVAL_BATCH: int      = 32     # match train batch
GRAD_ACCUM: int      = 1      # H100: 1, L40S: 2
LR: float            = 2.5e-4 # H100 with batch 32, scale up from 2e-4
EVAL_STEPS: int      = 500    # checkpoint frequency
NUM_CHECKPOINT_LIMIT = 2      # how many checkpoints to keep on disk
```

### Attention LoRA toggle — `gemini.py`

```python
moe_config.use_attention_lora = True   # True = attn ON, False = attn OFF
```

---

## Training

### Step 1 — Generate dataset

```bash
python main.py
```

This streams data from HuggingFace and saves to `data/train` and `data/eval`.
Takes 15-30 minutes depending on sample count. Will look stuck — it isn't.

### Step 2 — Launch training

```bash
nohup python gemini.py > training_30k_3ep_attn_on.log 2>&1 &
tail -f training_30k_3ep_attn_on.log
```

Name the log file to match what you're running so results stay organized.

### Step 3 — Keep studio alive (browser terminal)

```bash
while true; do echo "keepalive $(date)" >> keepalive.log; sleep 240; done
```

Run this in the Lightning AI browser terminal to prevent studio sleep.

### Step 4 — Monitor training

Watch logs:
```bash
tail -f training_30k_3ep_attn_on.log
```

Watch VRAM (separate terminal):
```bash
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Healthy training looks like:
