import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        if bidirectional:
            self.linear2 = Linear(d_model*2*2, d_model)
        else:
            self.linear2 = Linear(d_model*2, d_model)


        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_norm = self.norm3(src)
        #src_norm = src
        src2 = self.self_attn(src_norm, src_norm, src_norm, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class AIA_Transformer_Res(nn.Module):
    """
    Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size,output_size, dropout=0, num_layers=1):
        super(AIA_Transformer_Res, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            self.col_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=False))
            self.row_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size//2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        output_list = []
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b*dim2, -1)  # [F, B*T, c]
            row_output = self.row_trans[i](row_input)  # [F, B*T, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [B, C, T, F]
            row_output = self.row_norm[i](row_output)  # [B, C, T, F]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b*dim1, -1)
            col_output = self.col_trans[i](col_input)
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = output + self.k1*row_output + self.k2*col_output
            output_i = self.output(output)
            output_list.append(output_i)

        del row_input, row_output, col_input, col_output

        return output_i, output_list


class DenseEncoder_Res(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder_Res, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)

    def forward(self, x):
        enc_list = []
        x1 = self.conv_1(x)
        enc_list.append(x1)

        x2 = self.conv_2(x1)
        enc_list.append(x2)

        x3 = self.dilated_dense(x2)
        enc_list.append(x3)
        return x3, enc_list

class ComplexDecoder_Res_new(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder_Res_new, self).__init__()

        # self.sub_pixel = SPConvTranspose2d(num_channel, num_channel//2, (1, 3), 2)

        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)

        self.Trans_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_channel, out_channels=num_channel, kernel_size=(1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel),
        )

        self.Trans_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_channel, out_channels=2, kernel_size=(1, 3), stride=(1, 2)),

        )


    def forward(self, x):
        dec_list = []

        x1 = self.dense_block(x)
        dec_list.append(x1)

        x2 = self.Trans_conv_1(x1)
        dec_list.append(x2)

        x3 = self.Trans_conv_2(x2)

        x3 = x3[:, :, :, :-2]
        dec_list.append(x3)

        return x3, dec_list

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class AIA_Transformer_onelayer(nn.Module):
    """
    Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size,output_size, dropout=0):
        super(AIA_Transformer_onelayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True)
        self.col_trans = TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=False)
        self.row_norm = nn.GroupNorm(1, input_size//2, eps=1e-8)
        self.col_norm = nn.GroupNorm(1, input_size//2, eps=1e-8)


        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size//2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape

        output = self.input(input)

        row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b*dim2, -1)  # [F, B*T, c]
        row_output = self.row_trans(row_input)  # [F, B*T, c]
        row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [B, C, T, F]
        row_output = self.row_norm(row_output)  # [B, C, T, F]

        col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b*dim1, -1)
        col_output = self.col_trans(col_input)
        col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
        col_output = self.col_norm(col_output)
        output_single = output + self.k1*row_output + self.k2*col_output

        output_i = self.output(output_single)
        del row_input, row_output, col_input, col_output

        return output_i

class DPDCRN_Tea(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense_encoder = DenseEncoder_Res(in_channel=2, channels=128)

        self.dual_trans = AIA_Transformer_Res(128, 128, num_layers=4)

        self.complex_decoder = ComplexDecoder_Res_new(num_channel=128)


    def forward(
        self, spec
    ):

        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape

        out_1, enc_list = self.dense_encoder(spec)

        x_last, mid_out_list = self.dual_trans(out_1) #BCTF, #BCTFG


        complex_out, dec_list = self.complex_decoder(x_last)


        mask_real = complex_out[:, 0, :, :]
        mask_imag = complex_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]


        spec_mags = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
        spec_phase = torch.atan2(noisy_imag, noisy_real)

        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase,
            real_phase
        )
        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        enh_real = est_mags * torch.cos(est_phase)
        enh_imag = est_mags * torch.sin(est_phase)


        return enh_real, enh_imag, mid_out_list, enc_list, dec_list


class DPDCRN_Stu_M(nn.Module):
    def __init__(self):
        super().__init__()


        self.dense_encoder = DenseEncoder_Res(in_channel=2, channels=64)

        self.dual_trans = AIA_Transformer_onelayer(64, 64)

        self.complex_decoder = ComplexDecoder_Res_new(num_channel=64)


    def forward(
        self, spec
    ):

        # spec: [B, 2, T, Fc]
        b, c, t, f = spec.shape

        out_1, enc_list = self.dense_encoder(spec)

        x_last = self.dual_trans(out_1) #BCTF, #BCTFG

        complex_out, dec_list = self.complex_decoder(x_last)
        # mask_out = convT5_out[:, :, :, :-2]

        mask_real = complex_out[:, 0, :, :]
        mask_imag = complex_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        spec_mags = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
        spec_phase = torch.atan2(noisy_imag, noisy_real)

        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase,
            real_phase
        )
        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        enh_real = est_mags * torch.cos(est_phase)
        enh_imag = est_mags * torch.sin(est_phase)


        return enh_real, enh_imag, x_last, enc_list, dec_list

if __name__ == '__main__':

    input_tensor = torch.randn(1, 2, 100, 257)
    model = DPDCRN_Stu_M()
    enh_real, enh_imag, mid_list, enc_list, dec_list = model(input_tensor)
    print(enh_real.shape)
