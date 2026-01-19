# **手撕Transformer：从零开始详细讲解实现**

Transformer是现代 NLP 和多模态模型的基础（如 BERT、GPT、ViT 等），来自 2017 年的论文《Attention is All You Need》，核心思想是用自注意力（Self-Attention）机制取代 RNN 的序列依赖，实现并行计算。

今天我带你一步步“手撕”一个完整的 Transformer 模型。我们用 PyTorch 实现一个简单的序列到序列（Seq2Seq）的 Transformer，用于机器翻译任务（比如英-德翻译）。我会从整体架构开始，逐步拆解每个组件，解释原理、公式和代码。

**注意事项：**
- 我们实现的是一个简化版：Encoder 和 Decoder 各有 N=6 层，d_model=512，num_heads=8。
- 输入是序列 token（词向量），输出是翻译序列。
- 适合对象：
  - 已有 PyTorch 基础
  - 理解线性层 / softmax / embedding 等基础组件
  - 想从零写出一个可运行 Transformer，而不是只看 API

## 1. 整体架构概述
Transformer 主要分为两个部分：
- **Encoder**：处理输入序列，输出上下文表示。内部是多层 Encoder Layer，每层包括 Multi-Head Self-Attention + Feed Forward + Layer Norm + Residual Connections；
- **Decoder**：处理目标序列，输出预测。内部是多层 Decoder Layer，每层包括 Masked Multi-Head Self-Attention（自注意力，防止看到未来） + Multi-Head Encoder-Decoder Attention（交叉注意力） + Feed Forward + Layer Norm + Residual Connections。

额外组件：
- **Embedding**：词嵌入 + 位置编码（Positional Encoding）;
- **Output Layer**：线性层。

架构图：

![](images/Encoder-decoder-architecture-of-the-Transformer.jpg)

```
# 架构文字版
Input Embedding + Positional Encoding
↓
Encoder (N layers):
  - Multi-Head Self-Attention + Add & Norm
  ↓
  - Feed Forward + Add & Norm
↓
Decoder (N layers):
  - Masked Multi-Head Self-Attention + Add & Norm
  ↓
  - Multi-Head Encoder-Decoder Cross-Attention + Add & Norm
  ↓
  - Feed Forward + Add & Norm
↓
Linear
```



## 2. 输入嵌入（Embedding）和位置编码（Positional Encoding）

Transformer 不像 RNN 有顺序信息，所以需要位置编码来注入序列位置信息。

**原理：**
- 词嵌入：将 token ID 映射到 d_model 维向量。

- 位置编码：使用正弦/余弦函数生成位置向量，公式：
  
  $$
  PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
  $$

  $$
  PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
  $$

  其中 pos 是位置，i 是维度索引。正弦/余弦位置编码本质是**绝对位置编码**，但由于其函数形式具有可加性，模型在一定程度上可以通过线性组合间接推断相对位置信息；但这并非显式的相对位置建模。后续如 **RoPE、ALiBi** 才是明确引入相对位置信息的设计。

**代码实现：**

```python
import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x): # x.shape = (batch_size, seq_len)
        # Make the input scale more reasonable
        # avoid self.embed(x) too small
        return self.embed(x) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        posIdx = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(posIdx * div_term)
        pe[:, 1::2] = torch.cos(posIdx * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x): # x.shape = (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

**解释：**
- Embeddings：简单 nn.Embedding，乘 sqrt(d_model) 是论文中的技巧，避免嵌入太小。
- PositionalEncoding：预计算 pe 矩阵，加到输入上。max_len 是最大序列长度。

使用时：

```python
embed_layer = Embeddings(vocab_size, d_model)
pos_enc = PositionalEncoding(d_model)
input_emb = pos_enc(embed_layer(input_tokens))
```



## 3. 多头注意力（Multi-Head Attention）

这是 Transformer 的核心，也是理解整个模型性能与能力的关键部分。如果只精读一节，建议重点看这一节。

**原理：**
- 缩放点积注意力（Scaled Dot-Product Attention）：
  - 输入：Query (Q), Key (K), Value (V)，形状 (batch, num_heads, seq_len, d_k)；
  
  - 计算：
  
    $$
    Attention = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    
  
    其中 d_k = d_model / num_heads，除以 $sqrt(d_k)$ 是为了防止点积随维度增大而方差过大，导致 softmax 输出分布过于尖锐、训练不稳定。
  
- 多头：将 d_model 分成 num_heads 个头，每个头独立计算，然后 concat 并线性投影。

- 本文中 mask 为 bool 类型，True 表示该位置可参与注意力计算，False 表示被屏蔽。

- mask 种类：

  * Encoder 中的 **Self-Attention**： 用 padding mask（忽略 padding tokens）；
  * Decoder 中的 **Masked Self-Attention**：causal mask（上三角掩码，防止看到未来词） + padding mask；
  * Decoder 中的 **Encoder-Decoder Cross-Attention**：Cross-Attention 只需要对 **Encoder 侧的 padding 位置进行 mask**，使用 padding mask 即可，不需要 causal mask，因为 Decoder 在时间维度上的因果性已经在前一层 Masked self-attention 中得到保证。

* mask 形状：
  * 应可 broadcast 到 `(batch_size, num_heads, seq_len, seq_len)`；
  * 通常 padding mask 形状为 `(batch_size, 1, 1, seq_len)`；
  * causal mask 形状为 `(1, 1, seq_len, seq_len)`，可与 padding mask 逐元素的逻辑与（`padding_mask & causal_mask`）。

**代码实现：**

```python
def scaled_dot_product_attention(q, k, v, mask = None, dropout_layer = None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k) # (batch_size, num_heads, seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weight = torch.softmax(scores, dim=-1)

    if dropout_layer is not None:
        weight = dropout_layer(weight)

    return torch.matmul(weight, v) # (batch_size, num_heads, seq_len, d_k)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask = None): # (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        # (batch_size, num_heads, seq_len, d_k)
        q = self.linear[0](q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear[1](k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear[2](v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
     
        x = scaled_dot_product_attention(q, k, v, mask, self.dropout).transpose(1, 2) # (batch_size, seq_len, num_heads, d_k)
        x = x.contiguous().view(batch_size, -1, self.num_heads * self.d_k) # (batch_size, seq_len, d_model)

        return self.output_linear(x)
```

**解释：**

- self.linear：三个线性层分别投影 Q, K, V；
- view 和 transpose：将维度分成多头，便于并行计算；
- Dropout：在注意力权重上应用，防止过拟合，尽管新一代大模型已经很少使用 dropout，但在一些小模型或数据规模有限的场景中，dropout 依然有其价值；
- 实际工程中，在半精度训练时， `scores = scores.masked_fill(mask == 0, float('-inf'))` 这里有时会用一个较大的负数（如 `-1e9`）代替 `-inf`，以避免 softmax 数值异常。



## 4. 前馈网络（Feed Forward Network）
简单的前向全连接层。FFN 对序列中每个 token 独立、并行地应用相同的两层 MLP，不在 token 之间引入交互。因此，需要注意的是：**Attention 负责 token 间建模，FFN 负责 token 内建模**。

**原理：**

- 两层线性：第一层扩展到 d_ff（2048），ReLU 激活，然后压缩回 d_model。（现代模型中常常使用 GELU 和 SWiGLU 代替 ReLU）

- 公式：

  $$
  FFN(x) = max(0,xW1 + b1)W2+b2
  $$
  

**代码实现：**

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        return self.linear2(x)
```

**解释：** 简单两层 MLP，加 dropout 和 ReLU。



## 5. 层归一化（Layer Norm）和残差连接（Residual）
本文实现采用业界常用的 **Pre-LN** 结构，而非原论文中的 Post-LN 设计。前者在深层网络训练中具有更好的稳定性和收敛性，主要原因在于：

* Pre-LN 的关键优势不在于 LayerNorm 本身，而在于它恢复了**残差连接作为“恒等映射”的设计初衷**；

* 残差连接的核心作用，是在反向传播中提供一条**无条件的梯度高速通道**，使梯度可以在深层网络中稳定传播，而不依赖于子层的具体行为。

  然而，在 **Post-LN 结构中**：

  $$
  x_{l+1} = \mathrm{LN}(x_l + F(x_l))
  $$

  残差路径上的梯度**必须经过 LayerNorm 的 Jacobian**。

  这意味着：

  * 残差不再对应恒等映射；
  * 梯度会被 LayerNorm 的归一化、去均值和特征耦合所缩放和扭曲；
  * 残差路径从“无条件通道”退化为一个**依赖输入统计量的变换**；

  因此，Post-LN 并非“完全截断”梯度，而是**破坏了“恒等映射”这一残差网络最核心的结构假设**，在深层堆叠时导致梯度逐层衰减或不稳定。

* 相比之下，在 **Pre-LN** 结构中：

  $$
  x_{l+1} = x_l + F(\mathrm{LN}(x_l))
  $$

  残差分支在前向传播中保持为恒等映射，在反向传播中提供一条**不经过子层和归一化的直接梯度通道**：

  * 梯度可以绕过 LayerNorm 和子层直接传播；
  * 即使子层梯度退化，残差通道仍然提供 1:1 的梯度传递；
  * 深层网络的可训练性不再依赖于精细的初始化或学习率调度；

  因此，**Pre-LN 的稳定性和收敛性优势，源于其保留了残差连接作为“恒等梯度通道”的本质功能，而不是 LayerNorm 放在前面本身带来的数值技巧**。

**代码实现：** 

详见下文。



## 6. Encoder Layer
一个 Encoder 层：Self-Attention + FFN + Norm + Residual。

**代码实现：**
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layernorm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self_Attention + Residual + LayerNorm (Pre-LN)
        x_norm = self.layernorm[0](x)
        self_attn_output = self.self_attn(
            x_norm, 
            x_norm, 
            x_norm, 
            mask
        )
        x = x + self.dropout(self_attn_output)

        # FFN + Residual + LayerNorm (Pre-LN)
        x = x + self.dropout(self.feed_forward(self.layernorm[1](x)))

        return x
```

**解释：** 输入 x 是词嵌入 + 位置编码，mask 是 src_mask（padding mask）。



## 7. Decoder Layer
一个 Decoder 层：Masked Self-Attention + Encoder-Decoder Cross-Attention + FFN。

**代码实现：**
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layernorm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Masked Self-Attention + Residual + LayerNorm (Pre-LN)
        x_norm = self.layernorm[0](x)
        self_attn_output = self.self_attn(
            x_norm, 
            x_norm, 
            x_norm, 
            tgt_mask
        )  # tgt_mask is causal + padding
        x = x + self.dropout(self_attn_output)
        
        # Cross-Attention + Residual + LayerNorm (Pre-LN) 
        # memory is output of Encoder Layer
        cross_attn_output = self.cross_attn(
            self.layernorm[1](x), 
            memory, # it has been layernorm before being passed in
            memory, 
            src_mask
        )
        x = x + self.dropout(cross_attn_output)
        
        # FFN + Residual + LayerNorm (Pre-LN)
        x = x + self.dropout(self.feed_forward(self.layernorm[2](x)))
        return x
```

**解释：** memory 是 Encoder 的输出。tgt_mask 防止 Decoder 看到未来词。



## 8. 完整 Transformer 模型
堆叠 N 个 Layer。

**代码实现：**

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embed = Embeddings(src_vocab, d_model)
        self.tgt_embed = Embeddings(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        self.generator = nn.Linear(d_model, tgt_vocab)  # output layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Encoder
        src_emb = self.dropout(self.pos_enc(self.src_embed(src)))
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
        memory = self.encoder_norm(memory)
        
        # Decoder
        tgt_emb = self.dropout(self.pos_enc(self.tgt_embed(tgt)))
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, src_mask, tgt_mask)
        output = self.decoder_norm(output)
        
        # Generator
        logits = self.generator(output) # (batch, seq_len, tgt_vocab)
        return logits
```

**解释：**

- src/tgt：输入/目标 token，形状 (batch, seq_len);
- mask：由 Mask 生成函数生成;
- logits：实际训练时通常不显式调用 softmax，而是直接将 logits 送入 CrossEntropyLoss。



## 9. Mask 生成函数
- Padding Mask：忽略 padding（假设 padding_idx=0）。
- Causal Mask：Decoder 的上三角掩码。

**代码实现：**

```python
def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0 
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

# Decoder tgt_mask = causal_mask & padding_mask
```

**解释：** causal_mask 确保位置 i 只看到 <=i 的位置。



## 10. 完整代码和测试示例
把以上拼起来就是一个完整的 Transformer。

**完整代码**：见于 `./transformer-from-scratch.py`

测试示例：

```python
# copy all the classes and functions listed above over here

if __name__ == "__main__":
    # example parameters
    src_vocab = 10000
    tgt_vocab = 10000
    model = Transformer(src_vocab, tgt_vocab)

    # assuming input
    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 4
    src = torch.randint(1, src_vocab, (batch_size, src_seq_len)) 
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_seq_len))

    src_mask = create_padding_mask(src)
    tgt_mask = create_causal_mask(tgt_seq_len).to(tgt.device) & create_padding_mask(tgt)

    output = model(src, tgt, src_mask, tgt_mask)
    print(output.shape) # expected output: (2, 4, 10000)
```

**训练时**：用 teacher forcing（把正确答案右移一位喂给 decoder），让模型每步都看到正确的前缀，只预测下一个词，用 CrossEntropyLoss 计算损失； 

**生成（推理）时**：因为没有正确答案了，就只能靠自己从头一个词一个词地猜，所以用 greedy（最保险但容易陷入局部最优，容易重复、卡住）或者 beam search（质量更好但慢）来搜索最可能的序列。实际推理时通常还会对每一层缓存历史 K/V（KV cache），避免每一步对整个前缀重新计算 attention，从而将复杂度从 O(T²) 降到 O(T)，提高推理速度。



## 11. 结语

到这里，你已经从零实现了一个完整的 Encoder-Decoder Transformer :thumbsup:，包括：

- 位置编码
- 多头注意力
- Mask 机制
- Pre-LN 结构
- 端到端前向传播

现在你已经具备阅读和理解 BERT / GPT / ViT 源码的基础，其中 GPT 等 Decoder-only 模型可以看作是移除了 Encoder 和 Cross-Attention 的特例。