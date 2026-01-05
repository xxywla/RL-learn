import config
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass
import numpy as np
import datasets
import time

log_path = str(config.QWEN2_5_0_5B_SFT_LOG_PATH)


@dataclass
class SFTConfig:
    max_length: int = 2500
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    log_iter: int = 400
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 1000


def tokenize_and_format(data):
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=2500,
    )
    return input_ids


# 定义一个日志记录函数
def log_call(iters, iters_average_loss):
    with open(log_path, "a") as my_file:
        my_file.write(
            f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")},  iters:{iters + 1}, iters_average_Loss:{iters_average_loss:.4f}\n')


def linear_warmup(current_step, warmup_steps, max_lr):
    if current_step < warmup_steps:
        return max_lr * current_step / warmup_steps
    else:
        return max_lr


def cosine_decay(current_step, warmup_steps, total_steps, max_lr, min_lr):
    if current_step < warmup_steps:
        return linear_warmup(current_step, warmup_steps, max_lr)
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return (max_lr - min_lr) * decay + min_lr


def create_answer_mask(input_ids, tokenizer):
    """
    创建仅对助手回答部分计算损失的掩码

    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        tokenizer: 分词器

    Returns:
        answer_mask: 助手回答部分为1，其他部分为0的掩码
    """
    batch_size, seq_len = input_ids.shape
    answer_mask = torch.zeros_like(input_ids)

    # 获取结束标记的token id
    eos_token_id = tokenizer.encode('<|im_end|>')[0]

    for batch_idx in range(batch_size):
        # 找到所有 <|im_end|> 的位置
        eos_positions = torch.where(
            input_ids[batch_idx] == eos_token_id
        )[0].tolist()

        if len(eos_positions) < 2:  # 至少需要user和assistant各一个结束标记
            continue

        # 解析对话轮次
        user_ends, assistant_ends = \
            _parse_conversation_turns(eos_positions)

        # 为每个助手回答设置掩码
        _set_answer_masks(
            answer_mask[batch_idx],
            user_ends,
            assistant_ends,
            seq_len
        )

    return answer_mask


def _parse_conversation_turns(eos_positions):
    """
    解析对话轮次，分离用户和助手的结束位置

    对话格式：
    <|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n

    eos_positions[0]: system结束 (如果有)
    eos_positions[1]: 第1轮user结束
    eos_positions[2]: 第1轮assistant结束
    eos_positions[3]: 第2轮user结束
    eos_positions[4]: 第2轮assistant结束
    ...
    """
    # 跳过system系统提示词部分，从第一个user开始
    conversation_eos = eos_positions[1:]  # 去掉system的<im_end>

    # 偶数索引：user结束位置，奇数索引：assistant结束位置
    user_ends = [pos + 1 for pos in conversation_eos[::2]]  # 每隔2个取一个，从0开始
    assistant_ends = [pos + 1 for pos in conversation_eos[1::2]]  # 每隔2个取一个，从1开始

    return user_ends, assistant_ends


def _set_answer_masks(mask, user_ends, assistant_ends, seq_len):
    """
    为助手回答部分设置掩码

    Args:
        mask: 当前样本的掩码 [seq_len]
        user_ends: 用户消息结束位置列表
        assistant_ends: 助手消息结束位置列表
        seq_len: 序列长度
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)

    if num_user_turns == num_assistant_turns:
        # 完整对话：每轮都有用户问题和助手回答
        for user_end, assistant_end in zip(user_ends, assistant_ends):
            answer_start = user_end + 3  # 跳过 <|im_start|>assistant\n
            answer_end = assistant_end - 1  # 不包含 <|im_end|>
            mask[answer_start:answer_end] = 1

    elif num_user_turns == num_assistant_turns + 1:
        # 未完成对话：最后一轮助手回答被截断

        # 处理完整的对话轮次
        for user_end, assistant_end in zip(user_ends[:-1], assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1]
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1  # 到序列结尾


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = str(config.QWEN2_5_0_5B_PATH)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
    )
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(model.generation_config)

    model.generation_config.do_sample = True
    model.generation_config.eos_token_id = [151645, 151643]
    model.generation_config.pad_token_id = 151643
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.top_k = 20
    model.generation_config.repetition_penalty = 1.05

    print(model.generation_config)

    ultrachat_200k_data = datasets.load_dataset(str(config.ULTRA_CHAT_200K_PATH))
    print(ultrachat_200k_data["train_sft"][0])

    ## 生成训练数据的token id
    chosen_input_ids_list = []
    i = 0
    while True:
        data = ultrachat_200k_data['train_sft'][i]['messages']
        # 添加 **系统提示词**
        data.insert(
            0,
            {"content": "You are a helpful assistant", "role": "system"}
        )
        input_ids = tokenize_and_format(data)
        chosen_input_ids_list.append(input_ids)
        i += 1
        if i % 1000 == 0:
            print(f"已处理{i}条数据")
        if i == 50000:  # len(ultrachat_200k_data['train_sft']):
            break
    print('-' * 70)

    batch_size = SFTConfig.batch_size
    gradient_accumulation_steps = SFTConfig.gradient_accumulation_steps
    log_iter = SFTConfig.log_iter
    max_lr = SFTConfig.max_lr
    min_lr = SFTConfig.min_lr
    warmup_steps = SFTConfig.warmup_steps
    total_steps = len(chosen_input_ids_list) // batch_size
    optimizer = torch.optim.AdamW(filter(
        lambda p: p.requires_grad,
        model.parameters()
    ), lr=max_lr)
    trainable_parameters_num = sum(p.numel() for p in filter(
        lambda p: p.requires_grad,
        model.parameters()))
    print(f"训练参数数量：{trainable_parameters_num}")

    with open(log_path, "a") as my_file:
        my_file.write(
            f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")},batch_size:{batch_size},trainable_parameters_num:{trainable_parameters_num}, warmup_steps:{warmup_steps},max_lr:{max_lr}, min_lr:{min_lr}\n')

    model = model.to(device)
    model.train()
    training_losses = []
    model.zero_grad()  # 训练开始时清空梯度
    skipped_batches_count = 0

    total_batches = len(chosen_input_ids_list) // batch_size

    for batch_idx in range(total_batches):
        ## ==================== 数据准备阶段 ====================

        # 获取当前批次的原始数据
        current_batch_sequences = chosen_input_ids_list[
            batch_idx * batch_size: (batch_idx + 1) * batch_size
        ]

        # 计算当前批次的最大序列长度，用于padding对齐
        max_sequence_length = max([len(sequence) for sequence in current_batch_sequences])

        ### 对批次数据进行右填充，使所有序列长度一致以便并行计算
        padded_sequences_list = []
        pad_token_id = model.generation_config.eos_token_id[-1]

        for seq_idx in range(batch_size):
            # 原始的一条训练数据
            original_sequence = current_batch_sequences[seq_idx]
            # 要填充的长度
            padding_length = max_sequence_length - len(original_sequence)

            # 使用EOS token进行右填充
            padded_sequence = torch.nn.functional.pad(
                torch.tensor(original_sequence),
                (0, padding_length),
                mode='constant',
                value=pad_token_id
            ).tolist()

            padded_sequences_list.append(padded_sequence)

        # 转换为张量
        batch_input_tensor = torch.tensor(padded_sequences_list)

        ## ==================== 构建输入输出对 ====================

        # 构建因果语言模型的输入输出对：x->y（下一个词预测）
        model_inputs = batch_input_tensor[:, :-1].to(device)  # 输入：前n-1个token
        target_labels = batch_input_tensor[:, 1:].to(device)  # 标签：后n-1个token

        ## ==================== 构建训练掩码 ====================

        # 构建掩码矩阵来控制损失计算范围
        # 1. padding_mask：标识哪些位置是填充token（不计算损失）
        # 2. answer_mask：标识哪些位置是助手回答部分（只对回答计算损失）

        ### 【填充掩码】：非填充token为1，填充token为0
        ### padding_mask中的问题部分的掩码也是1
        padding_mask = torch.where(target_labels == pad_token_id, 0, 1)

        ### 【回答掩码】：只有助手回答部分为1，其他部分为0
        assistant_answer_mask = create_answer_mask(model_inputs, tokenizer)

        ### 【组合掩码】：同时满足"非填充"且"是回答部分"的token才计算损失
        ### 取出交集，就是真正要计算的回答部分
        final_loss_mask = (assistant_answer_mask & padding_mask)

        ## ==================== 批次有效性检查 ====================

        # 检查当前批次是否有效：如果某个样本的回答部分完全为空，则跳过该批次
        # 这种情况通常发生在问题过长导致回答部分被截断时
        tokens_per_sample = final_loss_mask.sum(dim=-1)  # 每个样本的有效回答token数
        min_answer_tokens = tokens_per_sample.min().item()  # 最少的有效token数

        if min_answer_tokens == 0:
            print(f'⚠️ 跳过第{batch_idx + 1}批次：回答部分数据不足')
            skipped_batches_count += 1
            continue  # 跳过当前批次

        ## ==================== 模型前向传播 ====================

        # 执行前向传播，获取模型预测的logits
        # [batch_size, seq_length, vocab_size]
        model_logits = model(model_inputs).logits

        ## ==================== 损失计算 ====================

        # 计算带掩码的交叉熵损失
        # 步骤：logits -> softmax -> log -> gather -> 负对数似然 -> 掩码过滤 -> 平均

        # 1. 计算每个token的负对数似然损失，
        # 形状：[batch_size, seq_len, vocab_size]
        log_probabilities = torch.log(torch.softmax(model_logits, dim=-1))
        # 使用真正的目标token取出vocab_size长度的数组中token对应的对数概率
        # 形状：[batch_size, seq_len]
        gathered_log_probs = torch.gather(
            log_probabilities,
            dim=-1,
            index=target_labels.unsqueeze(2)
        )
        negative_log_likelihood = gathered_log_probs * (-1)  # 负对数似然
        token_losses = negative_log_likelihood.squeeze(2)

        # 2. 应用掩码并计算每个样本的平均损失
        masked_token_losses = torch.mul(token_losses, final_loss_mask)
        sample_losses = masked_token_losses.sum(dim=-1) / final_loss_mask.sum(dim=-1)

        # 3. 计算批次平均损失并应用梯度累积
        batch_average_loss = torch.nanmean(sample_losses) / gradient_accumulation_steps

        ## ==================== 反向传播和优化 ====================

        # 反向传播计算梯度
        batch_average_loss.backward()

        # 动态调整学习率（余弦衰减 + 预热）
        current_learning_rate = cosine_decay(
            batch_idx,
            warmup_steps,
            total_steps,
            max_lr,
            min_lr
        )

        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_learning_rate

        # 梯度累积：只在累积步数达到或最后一个批次时更新权重
        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_final_batch = (batch_idx + 1) == total_batches

        if is_accumulation_step or is_final_batch:
            optimizer.step()  # 更新模型权重
            optimizer.zero_grad()  # 清空梯度缓存

        ## ==================== 训练日志记录 ====================

        # 记录当前批次的损失（还原梯度累积的缩放）
        actual_batch_loss = batch_average_loss.item() * gradient_accumulation_steps
        training_losses.append(actual_batch_loss)

        # 定期输出训练进度
        should_log = (batch_idx + 1) % log_iter == 0 or is_final_batch

        if should_log:
            # 计算最近几个批次的平均损失
            recent_losses = training_losses[-log_iter:]
            recent_average_loss = np.nanmean(recent_losses)

            # 输出训练状态
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f'⏰ 时间: {current_time} | '
                  f'📊 批次: {batch_idx + 1}/{total_batches} | '
                  f'📈 最近{len(recent_losses)}批次平均损失: {recent_average_loss:.4f} | '
                  f'🎯 学习率: {current_learning_rate:.2e}')

            # 调用外部日志记录函数
            log_call(batch_idx, recent_average_loss)

    ## ==================== 训练完成总结 ====================

    print("🎉 训练完成!")
    print(f'📊 训练统计:')
    print(f'   - 总批次数: {total_batches}')
    print(f'   - 跳过批次数: {skipped_batches_count}')
    print(f'   - 有效批次数: {total_batches - skipped_batches_count}')
    print(f'   - 最终平均损失: {np.nanmean(training_losses[-100:]):.4f}')

    if skipped_batches_count > 0:
        skip_ratio = skipped_batches_count / total_batches * 100
        print(f'⚠️ 跳过批次占比: {skip_ratio:.2f}%')
        if skip_ratio > 10:
            print('💡 建议: 跳过批次过多，考虑增加最大序列长度或优化数据预处理')

    model_sft_path = str(config.QWEN2_5_0_5B_SFT_PATH)
    model.save_pretrained(model_sft_path)
    tokenizer.save_pretrained(model_sft_path)
