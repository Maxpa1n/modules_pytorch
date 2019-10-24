import torch
import torch.nn as nn
import math


# 借鉴 https://github.com/huggingface/transformers 的写法
class SelfAttention(nn.Module):
    def __init__(self, seq_len, hidden_size, num_heads, dropout, output_attentions=True):
        super(SelfAttention, self).__init__()
        # hyperparameters
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_attentions = output_attentions
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads))
        self.att_size = hidden_size // num_heads
        # layer
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.att_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, X, attention_mask):
        # X [batch_size, seq_len, hidden_size,]
        # attention_mask [batch_size, num_heads ,seq_len, seq_len]
        query = self.wq(X)
        key = self.wk(X)
        value = self.wv(X)

        # multi heads
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.att_size)

        if attention_mask is not None:
            # mask 值取一个很大的负数  softmax后就是0
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.att_size*self.num_heads,)
        context = context.view(*new_context_shape)

        outputs = (context, attention_probs) if self.output_attentions else (context,)
        # tuple(context [batch_size, seq_len, hidden_size],att [batch_size, num_heads, seq_len, seq_len])
        # 可选择输出attention的值  self.output_attentions = True
        return outputs


if __name__ == '__main__':
    X = torch.randn(5, 15, 30)
    mask = torch.randn(5, 3, 15, 15)
    selfatt = SelfAttention(seq_len=15, hidden_size=30, num_heads=3, dropout=0.5)
    context, attention_probs = selfatt(X, mask)
    print(context.shape)
    print(attention_probs.shape)
