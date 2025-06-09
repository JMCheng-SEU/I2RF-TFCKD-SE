import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ABF_res(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF_res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.InstanceNorm2d(mid_channel, affine=True),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU()
        )

        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None

    def forward(self, x, y=None, shape = None):
        n, c, h, w = x.shape

        x = self.conv1(x)
        # x = self.BN1(x)
        # x = self.activ1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (h, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * (z[:,0].view(n,1,h,w).contiguous()) + y * (z[:,1].view(n,1,h,w).contiguous()))
        # output
        y = self.conv2(x)
        return y, x


class intra_fusion(nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channel, shapes, detach = False
    ):
        super(intra_fusion, self).__init__()
        self.shapes = shapes

        self.detach = detach

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF_res(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))

        self.abfs = abfs[::-1]

    def forward(self, features_set):

        if self.detach:
            for i in range(len(features_set)):
                features_set[i] = features_set[i].detach()

        x = features_set[::-1]
        out_features, res_features = self.abfs[0](x[0])
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)

        del res_features, x, features_set
        return out_features


def cosine_pairwise_similarities_perframe(features, eps=1e-6, normalized=True):
    features = features.permute(1, 0, 2)  ##### take timesteps as the first dim
    features_norm = torch.sqrt(torch.sum(features ** 2, dim=2, keepdim=True))
    features = features / (features_norm + eps)
    # features[features != features] = 0
    features_t = features.permute(0, 2, 1)
    similarities = torch.bmm(features, features_t)

    if normalized:
        similarities = (similarities + 1.0) / 2.0
    return similarities

class frame_MLPEmbed(nn.Module):

    def __init__(self, dim_in=1024, dim_out=128):
        super(frame_MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x

class freq_MLPEmbed(nn.Module):

    def __init__(self, dim_in=1024, dim_out=128):
        super(freq_MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x

class TF_calibration_block(nn.Module):

    def __init__(self, s_len, t_len, fea_dim, factor=4, detach = True):
        super(TF_calibration_block, self).__init__()

        self.l2norm = Normalize(2)

        self.detach = detach

        self.bsz = 8

        for i in range(t_len):
            setattr(self, 'time_key_weight' + str(i), frame_MLPEmbed(fea_dim, fea_dim))
            setattr(self, 'freq_key_weight' + str(i), freq_MLPEmbed(self.bsz, self.bsz * factor))
        for i in range(s_len):
            setattr(self, 'time_query_weight' + str(i), frame_MLPEmbed(fea_dim, fea_dim))
            setattr(self, 'freq_query_weight' + str(i), freq_MLPEmbed(self.bsz, self.bsz * factor))


    def forward(self, feat_s, feat_t):

        sim_t = list(range(len(feat_t)))
        sim_s = list(range(len(feat_s)))

        sim_t_g = list(range(len(feat_t)))
        sim_s_g = list(range(len(feat_s)))

        for i in range(len(feat_t)):
            BT, CT, T, DT = feat_t[i].shape
            if self.detach:
                feat_t[i] = feat_t[i].detach()

            sim_temp = feat_t[i].permute(0, 2, 1, 3)
            sim_temp = torch.reshape(sim_temp, [BT, T, CT * DT])

            sim_temp_t = sim_temp.permute(0, 2, 1)

            sim_t[i] = torch.bmm(sim_temp, sim_temp_t)

            sim_temp_g_1 = sim_temp.permute(1, 0, 2)
            sim_temp_g_2 = sim_temp.permute(1, 2, 0)
            sim_t_g[i] = (torch.matmul(sim_temp_g_1, sim_temp_g_2)).permute(1, 0, 2)



        for i in range(len(feat_s)):
            BS, CS, T, DS = feat_s[i].shape
            sim_temp = feat_s[i].permute(0, 2, 1, 3)

            sim_temp = torch.reshape(sim_temp, [BS, T, CS * DS])

            sim_temp_t = sim_temp.permute(0, 2, 1)

            sim_s[i] = torch.bmm(sim_temp, sim_temp_t)

            sim_temp_g_1 = sim_temp.permute(1, 0, 2)
            sim_temp_g_2 = sim_temp.permute(1, 2, 0)
            sim_s_g[i] = (torch.matmul(sim_temp_g_1, sim_temp_g_2)).permute(1, 0, 2)

        # key of target layers
        freq_proj_key = self.freq_key_weight0(sim_t_g[0])
        freq_proj_key = freq_proj_key[:, :, :, None]

        time_proj_key = self.time_key_weight0(sim_t[0])
        time_proj_key = time_proj_key[:, :, :, None]


        for i in range(1, len(sim_t)):
            temp_freq_proj_key = getattr(self, 'freq_key_weight' + str(i))(sim_t_g[i])
            freq_proj_key = torch.cat([freq_proj_key, temp_freq_proj_key[:, :, :, None]], 3)

            temp_time_proj_key = getattr(self, 'time_key_weight' + str(i))(sim_t[i])
            time_proj_key = torch.cat([time_proj_key, temp_time_proj_key[:, :, :, None]], 3)

        # query of source layers
        freq_proj_query = self.freq_query_weight0(sim_s_g[0])
        freq_proj_query = freq_proj_query[:, :, None, :]

        time_proj_query = self.time_query_weight0(sim_s[0])
        time_proj_query = time_proj_query[:, :, None, :]

        for i in range(1, len(sim_s)):
            temp_freq_proj_query = getattr(self, 'freq_query_weight' + str(i))(sim_s_g[i])
            freq_proj_query = torch.cat([freq_proj_query, temp_freq_proj_query[:, :, None, :]], 2)

            temp_time_proj_query = getattr(self, 'time_query_weight' + str(i))(sim_s[i])
            time_proj_query = torch.cat([time_proj_query, temp_time_proj_query[:, :, None, :]], 2)



        # freq attention
        freq_energy = torch.matmul(freq_proj_query, freq_proj_key)  # batch_size X No.stu feature X No.tea feature

        freq_attention = F.softmax(freq_energy, dim=-1)

        # time attention
        time_energy = torch.matmul(time_proj_query, time_proj_key)  # batch_size X No.stu feature X No.tea feature

        time_attention = F.softmax(time_energy, dim=-1)

        attention = []
        attention.append(time_attention)
        attention.append(freq_attention)

        # feature space alignment
        time_proj_value_stu = []
        time_value_tea = []

        freq_proj_value_stu = []
        freq_value_tea = []

        TF_proj_value_stu = []
        TF_value_tea = []
        for i in range(len(sim_s)):
            time_proj_value_stu.append([])
            time_value_tea.append([])

            freq_proj_value_stu.append([])
            freq_value_tea.append([])

            for j in range(len(sim_t)):

                BS, CS, T, DS = feat_s[i].shape
                BT, CT, T, DT = feat_t[j].shape
                feat_t[j] = feat_t[j].detach()

                att_feat_s = feat_s[i].permute(0, 2, 1, 3)
                att_feat_s = torch.reshape(att_feat_s, [BS, T, CS * DS])

                temp_feat_t = feat_t[j].permute(0, 2, 1, 3)
                temp_feat_t = torch.reshape(temp_feat_t, [BT, T, CT * DT])


                att_feat_s = att_feat_s.permute(1, 0, 2)
                temp_feat_t = temp_feat_t.permute(1, 0, 2)

                student_s_time = cosine_pairwise_similarities_perframe(att_feat_s)
                student_s_time = student_s_time / torch.sum(student_s_time, dim=2, keepdim=True)



                time_proj_value_stu[i].append(student_s_time)



                teacher_s_time = cosine_pairwise_similarities_perframe(temp_feat_t)
                teacher_s_time = teacher_s_time / torch.sum(teacher_s_time, dim=2, keepdim=True)


                time_value_tea[i].append(teacher_s_time)


                att_feat_s_freq = att_feat_s.permute(1, 0, 2)
                temp_feat_t_freq = temp_feat_t.permute(1, 0, 2)

                student_s_freq = cosine_pairwise_similarities_perframe(att_feat_s_freq)
                student_s_freq = student_s_freq / torch.sum(student_s_freq, dim=2, keepdim=True)
                student_s_freq = student_s_freq.permute(1, 0, 2)

                freq_proj_value_stu[i].append(student_s_freq)

                teacher_s_freq = cosine_pairwise_similarities_perframe(temp_feat_t_freq)
                teacher_s_freq = teacher_s_freq / torch.sum(teacher_s_freq, dim=2, keepdim=True)

                teacher_s_freq = teacher_s_freq.permute(1, 0, 2)

                freq_value_tea[i].append(teacher_s_freq)

        TF_proj_value_stu.append(time_proj_value_stu)
        TF_proj_value_stu.append(freq_proj_value_stu)

        TF_value_tea.append(time_value_tea)
        TF_value_tea.append(freq_value_tea)

        return TF_proj_value_stu, TF_value_tea, attention


class TF_SemCosineLoss(nn.Module):

    def __init__(self):
        super(TF_SemCosineLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, s_value, f_target, weight):

        time_s_value = s_value[0]
        freq_s_value = s_value[1]

        time_f_target = f_target[0]
        freq_f_target = f_target[1]

        time_weight = weight[0]
        freq_weight = weight[1]

        bsz, T, num_stu, num_tea = time_weight.shape
        time_ind_loss = torch.zeros(num_stu, num_tea).cuda()
        freq_ind_loss = torch.zeros(num_stu, num_tea).cuda()

        for i in range(num_stu):
            for j in range(num_tea):
                tmploss = (time_f_target[i][j] - time_s_value[i][j]) * (torch.log(time_f_target[i][j]) - torch.log(time_s_value[i][j]))

                tmp_weight = torch.unsqueeze(time_weight[:, :, i, j], 2)

                tmp_w_loss = torch.sum(tmploss * tmp_weight, dim=1)

                time_ind_loss[i, j] = (torch.mean(tmp_w_loss, dim=[0, 1]))

        for i in range(num_stu):
            for j in range(num_tea):
                tmploss = (freq_f_target[i][j] - freq_s_value[i][j]) * (torch.log(freq_f_target[i][j]) - torch.log(freq_s_value[i][j]))

                tmp_weight = torch.unsqueeze(freq_weight[:, :, i, j], 2)

                tmp_w_loss = torch.sum(tmploss * tmp_weight, dim=1)

                freq_ind_loss[i, j] = (torch.mean(tmp_w_loss, dim=[0, 1]))

        loss = ((time_ind_loss).sum()/(1.0*num_stu)) + ((freq_ind_loss).sum()/(1.0*num_stu))
        return loss


if __name__ == '__main__':

    mid_stu_fea = torch.randn(8, 64, 10, 64) ##### batch channel frame feature

    enc_stu_fea_list = []
    stu_enc_layer_1_out = torch.randn(8, 64, 10, 129)
    enc_stu_fea_list.append(stu_enc_layer_1_out)
    stu_enc_layer_2_out = torch.randn(8, 64, 10, 64)
    enc_stu_fea_list.append(stu_enc_layer_2_out)
    stu_enc_layer_3_out = torch.randn(8, 64, 10, 64)
    enc_stu_fea_list.append(stu_enc_layer_3_out)

    dec_stu_fea_list = []
    stu_dec_layer_1_out = torch.randn(8, 64, 10, 64)
    dec_stu_fea_list.append(stu_dec_layer_1_out)
    stu_dec_layer_2_out = torch.randn(8, 64, 10, 129)
    dec_stu_fea_list.append(stu_dec_layer_2_out)
    stu_dec_layer_3_out = torch.randn(8, 2, 10, 257)
    dec_stu_fea_list.append(stu_dec_layer_3_out)

    mid_tea_fea_list = []
    for index in range(4):
        temp_mid_tea_fea = torch.randn(8, 128, 10, 64)
        mid_tea_fea_list.append(temp_mid_tea_fea)

    enc_tea_fea_list = []
    tea_enc_layer_1_out = torch.randn(8, 128, 10, 129)
    enc_tea_fea_list.append(tea_enc_layer_1_out)
    tea_enc_layer_2_out = torch.randn(8, 128, 10, 64)
    enc_tea_fea_list.append(tea_enc_layer_2_out)
    tea_enc_layer_3_out = torch.randn(8, 128, 10, 64)
    enc_tea_fea_list.append(tea_enc_layer_3_out)

    dec_tea_fea_list = []
    tea_dec_layer_1_out = torch.randn(8, 128, 10, 64)
    dec_tea_fea_list.append(tea_dec_layer_1_out)
    tea_dec_layer_2_out = torch.randn(8, 128, 10, 129)
    dec_tea_fea_list.append(tea_dec_layer_2_out)
    tea_dec_layer_3_out = torch.randn(8, 2, 10, 257)
    dec_tea_fea_list.append(tea_dec_layer_3_out)

    criterion_kd = TF_SemCosineLoss()

    s_n_mid = [64, 64]
    t_n_mid = [128, 128, 128, 128]

    s_size = 10
    fea_dim = 8
    factor = 2



    self_attention_mid = TF_calibration_block(len(s_n_mid), len(t_n_mid), s_size, factor, True)

    s_n_enc = [64, 64, 64]
    t_n_enc = [128, 128, 128]


    self_attention_enc = TF_calibration_block(len(s_n_enc), len(t_n_enc), s_size, factor, True)

    s_n_dec = [64, 64, 2]
    t_n_dec = [128, 128, 2]

    self_attention_dec = TF_calibration_block(len(s_n_dec), len(t_n_dec), s_size, factor, True)

    s_n_dec = [64, 64, 64]
    t_n_dec = [128, 128, 128]

    self_attention_inter = TF_calibration_block(len(s_n_dec), len(t_n_dec), s_size, factor, False)

    mid_channel_stu = 128
    mid_channel_tea = 128


    ##### set mid intra fusion
    in_channels_mid_stu = [64]

    out_channels_mid_stu = [64]

    shapes_mid_stu = [64]

    fusion_mid_stu = intra_fusion(in_channels_mid_stu, out_channels_mid_stu, mid_channel_stu, shapes_mid_stu, False)


    in_channels_mid_tea = [128, 128, 128, 128]

    out_channels_mid_tea = [128, 128, 128, 128]

    shapes_mid_tea = [64, 64, 64, 64]

    fusion_mid_tea = intra_fusion(in_channels_mid_tea, out_channels_mid_tea, mid_channel_tea, shapes_mid_tea, True)


    ##### set encorder intra fusion
    in_channels_enc_stu = [64, 64, 64]

    out_channels_enc_stu = [64, 64, 64]

    shapes_enc_stu = [129, 64, 64]

    fusion_enc_stu = intra_fusion(in_channels_enc_stu, out_channels_enc_stu, mid_channel_stu, shapes_enc_stu, False)

    in_channels_enc_tea = [128, 128, 128]

    out_channels_enc_tea = [128, 128, 128]

    shapes_enc_tea = [129, 64, 64]

    fusion_enc_tea = intra_fusion(in_channels_enc_tea, out_channels_enc_tea, mid_channel_tea, shapes_enc_tea, True)



    ##### set decorder intra fusion
    in_channels_dec_stu = [64, 64, 2]
    out_channels_dec_stu = [64, 64, 2]
    shapes_dec_stu = [257, 129, 64]

    fusion_dec_stu = intra_fusion(in_channels_dec_stu, out_channels_dec_stu, mid_channel_stu, shapes_dec_stu, False)

    in_channels_dec_tea = [128, 128, 2]
    out_channels_dec_tea = [128, 128, 2]
    shapes_dec_tea = [257, 129, 64]

    fusion_dec_tea = intra_fusion(in_channels_dec_tea, out_channels_dec_tea, mid_channel_tea, shapes_dec_tea, True)


    ##### inter calibration KD

    tmp_mid_stu_fea = [mid_stu_fea]
    tmp_mid_tea_fea = mid_tea_fea_list[::-1]

    mid_fusion_out_stu = fusion_mid_stu(tmp_mid_stu_fea)
    mid_fusion_out_tea = fusion_mid_tea(tmp_mid_tea_fea)

    tmp_enc_stu_fea = enc_stu_fea_list[::-1]
    tmp_enc_tea_fea = enc_tea_fea_list[::-1]

    enc_fusion_out_stu = fusion_enc_stu(tmp_enc_stu_fea)
    enc_fusion_out_tea = fusion_enc_tea(tmp_enc_tea_fea)

    tmp_dec_stu_fea = dec_stu_fea_list
    tmp_dec_tea_fea = dec_tea_fea_list

    dec_fusion_out_stu = fusion_dec_stu(tmp_dec_stu_fea)
    dec_fusion_out_tea = fusion_dec_tea(tmp_dec_tea_fea)

    inter_stu_list = [enc_fusion_out_stu, mid_fusion_out_stu, dec_fusion_out_stu]
    inter_tea_list = [enc_fusion_out_tea, mid_fusion_out_tea, dec_fusion_out_tea]

    s_value, f_target, weight = self_attention_inter(inter_stu_list, inter_tea_list)
    inter_loss = criterion_kd(s_value, f_target, weight)

    ###### KD for Enc

    s_value, f_target, weight = self_attention_enc(enc_stu_fea_list, enc_tea_fea_list)
    enc_loss = criterion_kd(s_value, f_target, weight)

    ###### KD for Dec

    ###### self attention group transfer

    s_value, f_target, weight = self_attention_dec(dec_stu_fea_list, dec_tea_fea_list)
    dec_loss = criterion_kd(s_value, f_target, weight)

    ######## KD for Mid

    mid_stu_fea_list = [mid_stu_fea]

    s_value, f_target, weight = self_attention_mid(mid_stu_fea_list, mid_tea_fea_list)
    mid_loss = criterion_kd(s_value, f_target, weight)

    intra_loss = mid_loss + enc_loss + dec_loss
    KD_loss = intra_loss + inter_loss

    print(KD_loss)



