# forecast/models/AGCRN.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- AGCN.py 内容开始 ---
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k_idx in range(2, self.cheb_k): # Renamed loop variable to avoid conflict with cheb_k
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv
# --- AGCN.py 内容结束 ---


# --- AGCRNCell.py 内容开始 ---
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
# --- AGCRNCell.py 内容结束 ---


# --- AGCRN.py 内容开始 (已将 AGCRN 类改名为 Model) ---
class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class Model(nn.Module): # AGCRN 类已重命名为 Model
    def __init__(self, configs): # args 参数已重命名为 configs
        super(Model, self).__init__()
        # 参数映射：从 configs 对象获取 AGCRN 模型所需参数
        self.num_node = configs.enc_in # 假设 enc_in 是节点数量 (N)
        self.input_dim = 1 # 假设每个节点有一个输入特征维度 (D)
        self.hidden_dim = configs.d_model # 使用 d_model 作为隐藏维度 (rnn_units)
        self.output_dim = configs.c_out # configs.c_out 作为输出特征维度
        self.horizon = configs.pred_len # configs.pred_len 作为预测步长
        self.num_layers = configs.e_layers # 使用 e_layers 作为 AGCRN 的层数

        # AGCRN 特有参数，从 configs 获取或设定默认值
        # 你可能需要在 run.py 中为这些参数添加 argparse 定义
        self.cheb_k = getattr(configs, 'cheb_k', 3) # 默认 cheb_k 为 3
        self.embed_dim = getattr(configs, 'embed_dim', 16) # 默认 embed_dim 为 16
        self.default_graph = getattr(configs, 'default_graph', True) # 默认 default_graph 为 True

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_node, self.input_dim, self.hidden_dim, self.cheb_k,
                                self.embed_dim, self.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    # 调整 forward 方法签名以匹配 Exp_Basic
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: (B, T_input, D_input)
        # AGCRN 的 source 期望形状是 (B, T_input, N, D_feature)
        # 假设 x_enc 的 D_input 就是 N (num_nodes)，且每个节点特征维度是 1
        source = x_enc.unsqueeze(-1) # 转换为 (B, T_input, N, 1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T_input, N, hidden_dim
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden_dim (取最后一个时间步的输出)

        # CNN based predictor
        output = self.end_conv(output)                         #B, T_output*C_output, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T_output, N, C_output

        # 返回形状 [B, L, D]
        # 如果你的预测是针对所有节点的所有特征，则将其展平
        # 例如，将 (B, pred_len, N, C_out) 转换为 (B, pred_len, N*C_out)
        return output[:, -self.horizon:, :, :].reshape(output.shape[0], self.horizon, -1)
# --- AGCRN.py 内容结束 ---