import torch
import torch.nn as nn
import math

# ===================== Embedding and Positional Encoding =====================
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

# ========================== MultiHeadAttention ===============================
def scaled_dot_product_attention(q, k, v, mask = None, dropout_layer = None):
    d_model = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_model) # (batch_size, num_heads, seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weight = torch.softmax(scores, dim=-1)

    if dropout_layer is not None:
        weight = dropout_layer(weight)

    return torch.matmul(weight, v) # (batch_size, num_heads, seq_len, d_model)

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

# ========================= Feed Forward Network ==============================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        return self.linear2(x)
    
# ============================ Encoder Layer ==================================
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
    
# ============================ Decoder Layer ==================================
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

# ============================= Transformer ===================================
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
    
# ============================= Mask Generator ================================
def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0 
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

# Decoder tgt_mask = causal_mask & padding_mask

# ================================ Main =======================================
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