from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SST2_PATH = RAW_DATA_DIR / "sst2"

MODEL_DIR = Path(__file__).parent / "model"
GPT2_PATH = MODEL_DIR / "gpt2"
GPT2_SFT_PATH = MODEL_DIR / "gpt2-sft"
REWARD_MODEL_PATH = MODEL_DIR / "reward_model.pt"
PPO_MODEL_PATH = MODEL_DIR / "ppo_model.pt"

if __name__ == '__main__':
    print(GPT2_SFT_PATH)
