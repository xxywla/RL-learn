import torch
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM
from typing import Optional


class RewardHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # llm最后输出的隐藏层的维度
        self.hidden_size = config.hidden_size
        # 线性层用来对llm最后输出的隐藏层给奖励
        self.reward = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        # 使用正态分布初始化权重
        nn.init.normal_(
            self.reward.weight,
            std=(1.0 / np.sqrt(self.hidden_size + 1))
        )
        # 将偏置初始化为0
        nn.init.zeros_(self.reward.bias)

    def forward(self, hidden_states):
        # 给出奖励
        return self.reward(hidden_states)


class GPT2RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = RewardHead(self.llm.config)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]
        # 给出奖励
        reward = self.reward_head(last_hidden_state).squeeze(-1)
        # sigmoid用来将奖励搞到(-1,1)范围内
        return torch.sigmoid(reward)


class ValueHead(nn.Module):
    """
    ValueHead类为GPT2实现了一个“头”，会为输出的每个token返回一个标量值
    标量值就是这个token的价值，ValueHead就是评论家。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.value = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        nn.init.normal_(
            self.value.weight,
            std=(1.0 / np.sqrt(self.hidden_size + 1))
        )
        nn.init.zeros_(self.value.bias)

    def forward(self, hidden_states):
        output = hidden_states
        return self.value(output)


class ModelForCausalLMWithValueHead(nn.Module):
    """
    GPT2模型+一个价值头
    """

    def __init__(self, model_path):
        super().__init__()
        # 这个要初始化为我们微调出来的gpt2-sft模型
        # actor演员模型
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        # 添加价值头
        # critic评论家模型
        self.v_head = ValueHead(self.llm.config)

    def forward(
            self,
            input_ids,
            attention_mask,
    ) -> Optional[torch.FloatTensor]:
        # gpt2-sft模型的输出
        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 输出的token的概率分布，维度为 `vocab_size`
        lm_logits = transformer_outputs.logits
        # 获取最后一层隐藏层
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # 评估token的价值
        value = self.v_head(last_hidden_state).squeeze(-1)
        # 返回输出的token的logits和token的价值
        return lm_logits, value

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)
