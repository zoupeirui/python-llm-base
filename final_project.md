# 🧠 多模态大语言模型 (VLM) 核心知识补充

## 1. 什么是 VLM (Vision-Language Model)？
VLM 是能同时理解图像和文字的大语言模型。你可以给它一张照片并问“图中有什么”，它会像人一样用自然语言回答你。

### 代表模型谱系
| 模型 | 机构 | 核心思路 | 参数规模 |
| --- | --- | --- | --- |
| LLaVA | 威斯康星大学 | 用简单的 MLP 投影层连接 CLIP 视觉编码器和 LLaMA 文本解码器 | 7B / 13B |
| Qwen-VL / Qwen2.5-VL | 阿里云 | 在 Qwen 大模型基础上内嵌动态分辨率的视觉编码器 | 2B / 7B / 72B |
| InternVL | 上海 AI Lab | 更强的视觉编码器 (InternViT) + 动态高分辨率处理 | 2B / 8B / 76B |
| GPT-4o | OpenAI | 闭源，原生多模态（不分塔） | 未知 |

## 2. VLM 的三段式架构
几乎所有开源 VLM 都遵循同一套“三段式流水线”：

```text
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Vision Encoder│───▶│  Projection Layer │───▶│  Language Model   │
│  (ViT/CLIP)   │    │  (MLP / Q-Former) │    │  (LLaMA/Qwen)    │
│  冻结 ❄️      │    │  可训练 🔥        │    │  LoRA微调 🔥      │
└──────────────┘    └──────────────────┘    └──────────────────┘
     图片特征              对齐/翻译              自然语言生成
```

- **Vision Encoder**：从图片中提取高维视觉特征。通常用 CLIP-ViT 或 SigLIP，且在微调时冻结它（因为它已经训练得很好了）。
- **Projection Layer**：将视觉特征的维度“翻译”成语言模型能听懂的维度。LLaVA 用的是最简单的 2 层 MLP。
- **Language Model (LLM)**：最终的文字生成引擎。在微调阶段，我们通过 LoRA 只训练其中极小比例的参数。

## 3. 为什么不直接 Full Fine-Tuning（全量微调）？
| 指标 | Full Fine-Tuning | LoRA |
| --- | --- | --- |
| 训练参数量 | 100%（数十亿个） | 0.1% ~ 1% |
| 显存需求 (7B 模型) | ≥ 80GB (A100) | ≤ 16GB (单卡可跑) |
| 训练速度 | 极慢 | 快 3-10 倍 |
| 灾难性遗忘风险 | 高 | 极低 |
| 效果 | 略优 | 接近全量（通常 95%+） |

结论：在学术界和工业界的日常工作中，LoRA 微调已经成为绝对的主流。

## 4. LoRA (Low-Rank Adaptation) 的核心原理
LoRA 的核心思想用一句话概括：大模型的权重矩阵在微调时，变化量是低秩的（Low-Rank）。

原始权重矩阵 `W_0` 在微调后变成 `W_0 + ΔW`。LoRA 认为 `ΔW` 可以被分解为两个小矩阵的乘积：

```text
ΔW = B A
```

其中：

- `B ∈ R^{d×r}`
- `A ∈ R^{r×k}`
- `r` 是秩（Rank），通常取 8 或 16。

举例理解：

假设原始权重矩阵 `W` 的维度是 `4096×4096`，全量微调需要更新 `4096^2 = 16,777,216` 个参数。而 LoRA 只需要更新 `4096×8 + 8×4096 = 65,536` 个参数。参数量瞬间减少了 256 倍！

## 5. PEFT 库的关键配置参数
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # 秩 (Rank)：越大越强，但显存越多。一般 8 或 16 足够
    lora_alpha=16,          # 缩放系数：通常设为 2*r。实际缩放因子 = alpha/r
    lora_dropout=0.05,      # Dropout：防止过拟合
    target_modules=["q_proj", "v_proj"],  # 关键：选择注入 LoRA 的层
    bias="none",          # 不训练 bias，节省显存
    task_type="CAUSAL_LM" # 指定任务类型为自回归语言模型
)
```

### target_modules 的选择策略（极其重要）
- 保守策略：只注入 `q_proj` 和 `v_proj`（注意力层的 Query 和 Value 投影矩阵）。参数最少。
- 激进策略：注入 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`（所有线性层）。效果更好但显存翻倍。
- 科研界共识：至少注入 `q_proj` 和 `v_proj`，如果显卡够用就上全部线性层。

## 6. 多模态微调数据格式
VLM 微调数据通常是一个 JSON 文件，每条数据是一组图文对话：

```json
[
  {
    "image": "images/001.jpg",
    "conversations": [
      {"role": "user", "content": "<image>\nDescribe this image in detail."},
      {"role": "assistant", "content": "The image shows a golden retriever playing in a park..."}
    ]
  }
]
```

其中 `<image>` 是一个占位符，告诉模型“在这个位置插入图片的视觉 Token”。

## 7. 训练时的关键概念
- **梯度累积 (Gradient Accumulation)**：显存不够塞不下大 Batch？没关系。设置 `gradient_accumulation_steps=8`，它会连续做 8 个小 Batch 的前向传播，把梯度攒起来，最后一起更新一次参数。等效 Batch Size = 实际 Batch Size × 累积步数。
- **混合精度训练 (FP16 / BF16)**：将大部分计算从 32 位浮点降为 16 位，显存几乎减半，速度翻倍，精度几乎无损。在 Mac MPS 上可以用 fp16。
- **学习率调度 (LR Scheduler)**：通常用 cosine 余弦退火，学习率先 warmup 上升，然后缓慢下降。
- **DeepSpeed / FSDP**：多卡训练时的分布式框架。将模型参数分片到多张卡上。
