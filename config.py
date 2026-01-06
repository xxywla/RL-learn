from pathlib import Path

# 数据文件
DATA_DIR = Path(__file__).parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SST2_PATH = RAW_DATA_DIR / "sst2"

ULTRA_CHAT_200K_PATH = RAW_DATA_DIR / "ultrachat_200k"
ULTRA_FEEDBACK_PATH = RAW_DATA_DIR / "ultrafeedback_binarized"

COUNTDOWN_TASKS_PATH = RAW_DATA_DIR / "Countdown-Tasks-3to4"

# 模型文件
MODEL_DIR = Path(__file__).parent / "model"
GPT2_PATH = MODEL_DIR / "gpt2"
GPT2_SFT_PATH = MODEL_DIR / "gpt2-sft"
REWARD_MODEL_PATH = MODEL_DIR / "reward_model.pt"
PPO_MODEL_PATH = MODEL_DIR / "ppo_model.pt"

QWEN2_5_0_5B_PATH = MODEL_DIR / "Qwen2.5-0.5B"
QWEN2_5_0_5B_SFT_PATH = MODEL_DIR / "Qwen2.5-0.5B-SFT"
QWEN2_5_0_5B_DPO_PATH = MODEL_DIR / "Qwen2.5-0.5B-DPO"

QWEN2_5_3B_INSTRUCT_PATH = MODEL_DIR / "Qwen2.5-3B-Instruct"

# 日志文件
LOG_DIR = Path(__file__).parent / "log"

QWEN2_5_0_5B_SFT_LOG_PATH = LOG_DIR / "Qwen2.5-0.5B-SFT_log.txt"
QWEN2_5_0_5B_DPO_LOG_PATH = LOG_DIR / "Qwen2.5-0.5B-DPO_log.txt"

if __name__ == '__main__':
    print(GPT2_SFT_PATH)
