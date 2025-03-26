import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from modules.model.base import *
from modules.model.rnn import RnnDecoder
from modules.model.induced_att import ContinuousEncoding

from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

import os
from ..custom_transformers import TransformerEncoderLayer, TransformerDecoderLayer


def get_batch_mask(B, L, valid_len):
    mask = repeat(torch.arange(end=L, device=valid_len.device),
                  'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
    return mask


class PositionalEncoding(nn.Module):
    """
    A type of trigonometric encoding for indicating items' positions in sequences.
    """

    def __init__(self, embed_size, max_len):
        super().__init__()

        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """
        Args:
            x: (B, T, d_model)
            position_ids: (B, T) or None

        Returns:
            (1, T, d_model) / (B, T, d_model)
        """
        if position_ids is None:
            return self.pe[:, :x.size(1)]
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]  # (1, T, d_model)
            pe = pe.expand((position_ids.shape[0], -1, -1))  # (B, T, d_model)
            pe = pe.reshape(-1, self.d_model)  # (B * T, d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B * T,)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe


class TransformerEncoder(Encoder):
    """
    A basic Transformer Encoder.
    """
    def __init__(self, d_model, output_size,
                 sampler, dis_feats=[], num_embeds=[], con_feats=[],
                 num_heads=8, num_layers=2, hidden_size=128):
        super().__init__(sampler, 'Transformer-' +
                         ','.join(map(str, dis_feats + con_feats)) +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}')

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.output_size = output_size

        self.pos_encode = PositionalEncoding(d_model, max_len=2000)
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            self.con_linear = nn.Linear(len(con_feats), d_model)
        else:
            self.con_linear = None

        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

        encoder_layer = TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, trip, valid_len, **kwargs):
        B, L, E_in = trip.shape

        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, src_mask = self.sampler(trip, src_mask)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())  # (B, L, E)
        if self.con_linear is not None:
            h += self.con_linear(x[..., self.con_feats])
        h += self.pos_encode(h)

        memory = self.encoder(h, src_key_padding_mask=src_mask)  # (B, L, E)
        memory = torch.nan_to_num(memory)
        mask_expanded = repeat(src_mask, 'B L -> B L E', E=memory.size(2))  # (B, L, E)
        memory = memory.masked_fill(mask_expanded, 0)  # (B, L, E)
        memory = torch.sum(memory, 1) / valid_len.unsqueeze(-1)
        memory = self.out_linear(memory)  # (B, E_out) or (B, L, E_out)
        return memory
    
class TribleConv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size_list=None, stride=1):
        super(TribleConv, self).__init__()

        if kernel_size_list is None:
            kernel_size_list = [3, 5, 7]
        padding_list = [kernel_size // 2 for kernel_size in kernel_size_list]

        self.input_size = input_size
        self.output_size = output_size

        # 检查三个output_size是否是3的倍数
        if output_size % 3 != 0:
            raise ValueError("output_size must be a multiple of 3.")

        self.out_channels = output_size // 3

        self.conv1 = nn.Conv1d(in_channels=self.input_size, out_channels=self.out_channels, kernel_size=kernel_size_list[0], stride=stride, padding=padding_list[0])
        self.conv2 = nn.Conv1d(in_channels=self.input_size, out_channels=self.out_channels, kernel_size=kernel_size_list[1], stride=stride, padding=padding_list[1])
        self.conv3 = nn.Conv1d(in_channels=self.input_size, out_channels=self.out_channels, kernel_size=kernel_size_list[2], stride=stride, padding=padding_list[2])

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        x = x.transpose(1, 2)
        # x: [batch_size, input_size, seq_len]
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        # outN: [batch_size, out_channels, seq_len]
        out = torch.cat([out1, out2, out3], dim=1)
        # out: [batch_size, output_size, seq_len]
        out = out.transpose(1, 2)
        # out: [batch_size, seq_len, output_size]
        return out

# 定义BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()

        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)

        # 全连接层，将LSTM的输出映射到最终的输出
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 乘以2是因为双向

    def forward(self, x):
        # LSTM的输出
        lstm_out, _ = self.lstm(x)

        # 全连接层输出
        out = self.fc(lstm_out)
        return out

class LlamaMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LlamaMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x


class LlamaEncoder(Encoder):
    """
    A basic Transformer Encoder.
    """
    def __init__(self, d_model, output_size,
                 sampler, dis_feats=[], num_embeds=[], con_feats=[],
                 num_heads=8, num_layers=2, hidden_size=128):
        super().__init__(sampler, 'LlamaEncoder')

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.output_size = output_size

        self.pos_encode = PositionalEncoding(d_model, max_len=2000)
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            self.con_linear = nn.Linear(len(con_feats), d_model)
        else:
            self.con_linear = None

        # 引入卷积与LSTM
        self.conv = TribleConv(input_size=d_model, output_size=3*d_model)
        self.conv_adapter = nn.Linear(3*d_model, d_model)
        self.LSTM = BiLSTM(input_size=d_model, hidden_size=128, output_size=d_model, num_layers=4, dropout=0.25)
        self.fuser = nn.Linear(d_model*2, d_model)

        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

        encoder_layer = TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, trip, valid_len, **kwargs):
        B, L, E_in = trip.shape

        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, src_mask = self.sampler(trip, src_mask)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())  # (B, L, E)
        if self.con_linear is not None:
            h += self.con_linear(x[..., self.con_feats])
        h += self.pos_encode(h)

        # 进入卷积与LSTM提取
        masked_trips1 = self.LSTM(h)
        masked_trips2 = self.conv(h)
        masked_trips2 = self.conv_adapter(masked_trips2)
        trips_emb = torch.cat([masked_trips1, masked_trips2], dim=2)
        h = self.fuser(F.gelu(trips_emb))

        memory = self.encoder(h, src_key_padding_mask=src_mask)  # (B, L, E)
        memory = torch.nan_to_num(memory)
        mask_expanded = repeat(src_mask, 'B L -> B L E', E=memory.size(2))  # (B, L, E)
        memory = memory.masked_fill(mask_expanded, 0)  # (B, L, E)
        memory = torch.sum(memory, 1) / valid_len.unsqueeze(-1)
        memory = self.out_linear(memory)  # (B, E_out) or (B, L, E_out)
        return memory
    
class LlamaEncoder2(Encoder):
    """
    A basic Transformer Encoder.
    """
    def __init__(self, d_model, output_size,
                 sampler, model_path, dis_feats=[], num_embeds=[], con_feats=[], 
                 num_heads=8, num_layers=2, hidden_size=128):
        super().__init__(sampler, 'LlamaEncoder')

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.output_size = output_size

        self.pos_encode = PositionalEncoding(d_model, max_len=2000)
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            self.con_linear = nn.Linear(len(con_feats), d_model)
        else:
            self.con_linear = None

        self.model_path = model_path
        # Llama model
        self.useLlama = True
        useLora = True
        if self.useLlama:
            self.lora_config = LoraConfig(r=16,
                                        lora_alpha=32,
                                        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                                        lora_dropout=0.05,
                                        bias="none",
                                        task_type="CAUSAL_LM")

            self.llama = LlamaForCausalLM.from_pretrained(os.path.join(os.environ['MODEL_PATH'], self.model_path))
            self.hidden_dim = self.llama.config.hidden_size

            if useLora:
                self.llama = get_peft_model(self.llama, self.lora_config)
            else:
                for param in self.llama.parameters():
                    param.requires_grad = False
        else:
            self.llama = nn.Identity()
            self.hidden_dim = 2048

        # 引入卷积与LSTM
        self.conv = TribleConv(input_size=d_model, output_size=3*d_model)
        self.conv_adapter = nn.Linear(3*d_model, d_model)
        self.LSTM = BiLSTM(input_size=d_model, hidden_size=128, output_size=d_model, num_layers=4, dropout=0.25)
        self.fuser = nn.Linear(d_model*2, self.hidden_dim)
        self.llama_adapter = nn.Linear(self.hidden_dim, d_model)

        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

        encoder_layer = TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, trip, valid_len, **kwargs):
        B, L, E_in = trip.shape

        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, src_mask = self.sampler(trip, src_mask)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())  # (B, L, E)
        if self.con_linear is not None:
            h += self.con_linear(x[..., self.con_feats])
        h += self.pos_encode(h)

        # 进入卷积与LSTM提取
        masked_trips1 = self.LSTM(h)
        masked_trips2 = self.conv(h)
        masked_trips2 = self.conv_adapter(masked_trips2)
        trips_emb = torch.cat([masked_trips1, masked_trips2], dim=2)
        h = self.fuser(F.gelu(trips_emb))

        if self.useLlama:
            # llama inference
            outputs = self.llama(
                inputs_embeds=h,
                output_hidden_states=True
            )
            h = outputs.hidden_states[-1]
        h = self.llama_adapter(h)

        memory = self.encoder(h, src_key_padding_mask=src_mask)  # (B, L, E)
        memory = torch.nan_to_num(memory)
        mask_expanded = repeat(src_mask, 'B L -> B L E', E=memory.size(2))  # (B, L, E)
        memory = memory.masked_fill(mask_expanded, 0)  # (B, L, E)
        memory = torch.sum(memory, 1) / valid_len.unsqueeze(-1)
        memory = self.out_linear(memory)  # (B, E_out) or (B, L, E_out)
        return memory


class LlamaEncoder3(Encoder):
    def __init__(self, input_size, output_size, model_path, device, sampler, dis_feats=[], num_embeds=[], con_feats=[], version=2):
        super().__init__(sampler, f"Llama-v{version}")
        
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.model_path = model_path


        # Llama model
        self.useLlama = False
        useLora = True
        if self.useLlama:
            self.lora_config = LoraConfig(r=16,
                                        lora_alpha=32,
                                        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                                        lora_dropout=0.05,
                                        bias="none",
                                        task_type="CAUSAL_LM")

            self.llama = LlamaForCausalLM.from_pretrained(os.path.join(os.environ['MODEL_PATH'], self.model_path))
            self.hidden_dim = self.llama.config.hidden_size

            if useLora:
                self.llama = get_peft_model(self.llama, self.lora_config)
            else:
                for param in self.llama.parameters():
                    param.requires_grad = False
        else:
            self.llama = nn.Identity()
            self.hidden_dim = 2048

        

        # 分别处理离散特征和连续特征
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.adapt_size = 64

        self.pos_encode = PositionalEncoding(self.adapt_size, max_len=2000)

        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, self.adapt_size) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            self.con_linear = nn.Linear(len(con_feats), self.adapt_size)
        else:
            self.con_linear = None

        self.mix_size = self.adapt_size



        # 输入层
        self.conv = TribleConv(input_size=self.mix_size, output_size=384)
        self.conv_adapter = nn.Linear(384, self.hidden_dim//4)
        self.LSTM = BiLSTM(input_size=self.mix_size,hidden_size=128,output_size=self.hidden_dim//4,num_layers=4,dropout=0.25)
        self.fuser = nn.Linear(self.hidden_dim//2, self.hidden_dim)
        


        # 输出层
        self.final_LSTM = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim//2, num_layers=3, batch_first=True)
        self.final_adapter = nn.Linear(self.hidden_dim//2, self.output_size)

        #self.MLP = LlamaMLP(self.hidden_dim, self.hidden_dim//2 , self.output_size)

    def forward(self, trips, valid_lens, **kwargs):
        B, L, H = trips.shape  # 获取矩阵的形状

        # x = None
        # if self.dis_embeds is not None:
        #     for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
        #         if x is None:
        #             x = dis_embed(trips[..., dis_feat].long())
        #         else:
        #             x = torch.cat([x, dis_embed(trips[..., dis_feat].long())], dim=-1) # (B, L, E)
 

        # # print(x.shape)
        # if self.con_linear is not None:
        #     if x is None:
        #         x = self.con_linear(trips[..., self.con_feats])
        #     else:
        #         x = torch.cat([x, self.con_linear(trips[..., self.con_feats])], dim=-1)

        h = torch.zeros(B, trips.size(1), self.adapt_size).to(trips.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(trips[..., dis_feat].long())  # (B, L, E)
        if self.con_linear is not None:
            h += self.con_linear(trips[..., self.con_feats])
        h += self.pos_encode(h)

        # 生成每一行的索引 (0, 1, 2, ..., L-1) 并与 valid_len 进行比较
        valid_lens_tensor = valid_lens.unsqueeze(1)  # 将 valid_len 转换为 B x 1 的形状
        mask = torch.arange(L).unsqueeze(0).to(self.device) < valid_lens_tensor  # 生成 B x L 的布尔掩码
        mask = mask.unsqueeze(2).expand(-1, -1, self.mix_size)  # 扩展到 B x L x H 的形状
        masked_trips = h * mask  # 将无效部分掩码为0

        masked_trips1 = self.LSTM(masked_trips)

        masked_trips2 = self.conv(masked_trips)
        masked_trips2 = self.conv_adapter(masked_trips2)

        trips_emb = torch.cat([masked_trips1, masked_trips2], dim=2)

        trips_emb = self.fuser(F.gelu(trips_emb))

        if self.useLlama:
            # llama inference
            outputs = self.llama(
                inputs_embeds=trips_emb,
                output_hidden_states=True
            )
            logits = outputs.hidden_states[-1]
        
        else:
            logits = trips_emb

        # trips = self.MLP(logits)
        # trips = torch.mean(trips,dim=1)

        _, (h, _) = self.final_LSTM(logits)
        trips = h[-1]  # (B, hidden_size)

        trips = self.final_adapter(trips)


        return trips

class LlamaDecoder(Decoder):
    def __init__(self, encode_size, hidden_size, output_size, num_layers=2):
        super().__init__(f'LlamaDecoder')

        self.encode_size = encode_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 从潜在空间映射到初始隐藏状态
        self.latent_to_hidden = nn.Linear(encode_size, hidden_size * num_layers)
        
        # 解码器GRU
        self.gru = nn.GRU(
            output_size,  # 输入是上一步的输出
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 从隐藏状态映射到输出
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # 初始输入
        self.initial_input = nn.Parameter(torch.zeros(1, output_size))
        
    def forward(self, tgt, encode, teacher_forcing_ratio=0.5):
        """
        参数:
            tgt: 形状为 (batch_size, seq_len, input_size) 的目标序列
            encode: 形状为 (batch_size, encode_size) 的潜在向量
            teacher_forcing_ratio: 使用教师强制的概率
        返回:
            outputs: 形状为 (batch_size, max_length, output_size) 的重构轨迹
        """

        batch_size = encode.size(0)
        max_len = tgt.size(1)
        
        # 准备输出张量
        outputs = torch.zeros(batch_size, max_len, self.output_size).to(encode.device)
        
        # 从潜在向量初始化隐藏状态
        hidden = self.latent_to_hidden(encode)
        hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)
        
        # 准备初始输入
        decoder_input = self.initial_input.expand(batch_size, -1)
        
        # 逐步解码
        for t in range(max_len):
            # 单步解码

            output, hidden = self.gru(decoder_input.unsqueeze(1), hidden)
            output = self.fc_out(output.squeeze(1))
            outputs[:, t] = output
            
            # 决定下一步的输入（教师强制或使用自己的预测）
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force and tgt is not None:
                decoder_input = tgt[:, t]
            else:
                decoder_input = output
                
        return outputs


class TransformerDecoder(Decoder):
    def __init__(self, encode_size, d_model, hidden_size, num_layers, num_heads):
        super().__init__(f'TransDecoder-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}')
        self.d_model = d_model

        self.memory_linear = nn.Linear(encode_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, max_len=2000)
        self.start_token = nn.Parameter(torch.randn(d_model), requires_grad=True)

        layer = TransformerDecoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=hidden_size, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt, encode):
        memory = self.memory_linear(encode).unsqueeze(1)  # (B, 1, E)

        tgt_mask = self.gen_casual_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer(tgt + self.pos_encode(tgt), memory, tgt_mask=tgt_mask)
        return out

    @staticmethod
    def gen_casual_mask(seq_len, include_self=True):
        """
        Generate a casual mask which prevents i-th output element from
        depending on any input elements from "the future".
        Note that for PyTorch Transformer model, sequence mask should be
        filled with -inf for the masked positions, and 0.0 else.

        :param seq_len: length of sequence.
        :return: a casual mask, shape (seq_len, seq_len)
        """
        if include_self:
            mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
        else:
            mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
        return mask.bool()

