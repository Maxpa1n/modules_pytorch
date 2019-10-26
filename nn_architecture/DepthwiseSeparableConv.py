import torch
import torch.nn as nn


# from QAnet https://github.com/hengruo/QANet-pytorch
# input [batch, hid_dim, seq_len] for word
# inout [batch, hid_dim, word_len, seq_len] for char
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            # groups 减少了卷积核的个数,一个卷积核服用out_channels次
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


if __name__ == '__main__':
    x_word = torch.randn(5, 30, 15)
    conv_word = DepthwiseSeparableConv(in_ch=30, out_ch=15, k=3, dim=1)
    out_word = conv_word(x_word)
    print(out_word.shape)
    print(conv_word.depthwise_conv.weight.shape)