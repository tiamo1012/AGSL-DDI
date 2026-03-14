import torch
import torch.nn as nn
import torch.nn.functional as F

class D1Model(nn.Module):
    def __init__(self, config):
        super(D1Model, self).__init__()
        self.config = config
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.embedding_dim
        )
        
        # ================= Branch A: Global (BiLSTM) =================
        # 输出维度: hidden_dim * 2
        self.layer1_1 = nn.LSTM(
            input_size=config.embedding_dim, 
            hidden_size=config.hidden_dim_1d, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        if not config.shared:
            self.layer1_2 = nn.LSTM(
                input_size=config.embedding_dim, 
                hidden_size=config.hidden_dim_1d, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True
            )

        # ================= Branch B: Local (1D-CNN) =================
        # 输出维度: hidden_dim * 2 (与 LSTM 对齐)
        self.conv1_1 = nn.Conv1d(
            in_channels=config.embedding_dim, 
            out_channels=config.hidden_dim_1d * 2, 
            kernel_size=3, 
            padding=1
        )
        
        if not config.shared:
            self.conv1_2 = nn.Conv1d(
                in_channels=config.embedding_dim, 
                out_channels=config.hidden_dim_1d * 2, 
                kernel_size=3, 
                padding=1
            )

        # ================= Innovation 3: Gated Fusion Unit =================
        # 输入: LSTM(hidden*2) + CNN(hidden*2) = hidden*4
        # 输出: Gate系数 z (hidden*2)
        # 作用: 学习如何根据当前上下文动态分配 LSTM 和 CNN 的权重
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim_1d * 4, config.hidden_dim_1d * 2),
            nn.Sigmoid()
        )

        # ================= Interaction: Cross Attention =================
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim_1d * 2, 
            num_heads=4, 
            batch_first=True,
            dropout=config.dropout
        )

        # MLP Layers
        self.layer2_1 = nn.Linear(config.hidden_dim_1d * 2, config.hidden_dim_1d)
        if not config.shared: self.layer2_2 = nn.Linear(config.hidden_dim_1d * 2, config.hidden_dim_1d)
        
        self.layer3_1 = nn.Linear(config.hidden_dim_1d, 4)
        if not config.shared: self.layer3_2 = nn.Linear(config.hidden_dim_1d, 4)
        
        self.layer4 = nn.Linear(8, 1) 
        
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def cross_attention(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output

    def gated_fusion(self, lstm_out, cnn_out):
        """
        执行门控融合: z * LSTM + (1-z) * CNN
        """
        # 1. 拼接特征用来计算门控系数
        # shape: (Batch, Seq, Hidden*4)
        concat_feat = torch.cat([lstm_out, cnn_out], dim=-1)
        
        # 2. 计算门控系数 z (0~1之间)
        # shape: (Batch, Seq, Hidden*2)
        z = self.fusion_gate(concat_feat)
        
        # 3. 加权融合
        fusion_out = z * lstm_out + (1 - z) * cnn_out
        return fusion_out

    def forward(self, batch_data):
        ids1 = batch_data.ids1
        ids2 = batch_data.ids2
        
        current_batch_size = ids1.shape[0] // self.config.sequence_length

        # 1. Embedding
        emb1 = self.embedding(ids1.view(current_batch_size, self.config.sequence_length))
        emb2 = self.embedding(ids2.view(current_batch_size, self.config.sequence_length))

        if self.config.shared:
            # === Branch A: LSTM ===
            lstm_out1, _ = self.layer1_1(emb1)
            lstm_out2, _ = self.layer1_1(emb2)
            
            # === Branch B: CNN ===
            cnn_in1 = emb1.permute(0, 2, 1)
            cnn_in2 = emb2.permute(0, 2, 1)
            cnn_out1 = self.relu(self.conv1_1(cnn_in1)).permute(0, 2, 1)
            cnn_out2 = self.relu(self.conv1_1(cnn_in2)).permute(0, 2, 1)
            
            # === Innovation 3: Gated Fusion (关键修改) ===
            # 动态融合 LSTM 和 CNN
            seq_feat1 = self.gated_fusion(lstm_out1, cnn_out1)
            seq_feat2 = self.gated_fusion(lstm_out2, cnn_out2)
            
            # === Interaction: Cross Attention ===
            attn_out1 = self.cross_attention(seq_feat1, seq_feat2, seq_feat2)
            attn_out2 = self.cross_attention(seq_feat2, seq_feat1, seq_feat1)
            
            # Residual Connection
            seq_final1 = seq_feat1 + attn_out1
            seq_final2 = seq_feat2 + attn_out2
            
            # Max Pooling
            feat1, _ = torch.max(seq_final1, dim=1)
            feat2, _ = torch.max(seq_final2, dim=1)
            
            # MLP
            logits1 = self.dropout(self.relu(self.layer2_1(feat1)))
            logits2 = self.dropout(self.relu(self.layer2_1(feat2)))
            logits1 = self.dropout(self.relu(self.layer3_1(logits1)))
            logits2 = self.dropout(self.relu(self.layer3_1(logits2)))
            
        else:
            # Independent Mode (同理应用 Gated Fusion)
            lstm_out1, _ = self.layer1_1(emb1)
            lstm_out2, _ = self.layer1_2(emb2)
            
            cnn_in1 = emb1.permute(0, 2, 1)
            cnn_in2 = emb2.permute(0, 2, 1)
            cnn_out1 = self.relu(self.conv1_1(cnn_in1)).permute(0, 2, 1)
            cnn_out2 = self.relu(self.conv1_2(cnn_in2)).permute(0, 2, 1)
            
            # Gated Fusion
            seq_feat1 = self.gated_fusion(lstm_out1, cnn_out1)
            seq_feat2 = self.gated_fusion(lstm_out2, cnn_out2)
            
            attn_out1 = self.cross_attention(seq_feat1, seq_feat2, seq_feat2)
            attn_out2 = self.cross_attention(seq_feat2, seq_feat1, seq_feat1)
            
            seq_final1 = seq_feat1 + attn_out1
            seq_final2 = seq_feat2 + attn_out2
            
            feat1, _ = torch.max(seq_final1, dim=1)
            feat2, _ = torch.max(seq_final2, dim=1)
            
            logits1 = self.dropout(self.relu(self.layer2_1(feat1)))
            logits2 = self.dropout(self.relu(self.layer2_2(feat2)))
            logits1 = self.dropout(self.relu(self.layer3_1(logits1)))
            logits2 = self.dropout(self.relu(self.layer3_2(logits2)))

        combined_features = torch.cat((logits1, logits2), dim=1) 
        final_logits = self.layer4(combined_features)
        output = self.sigmoid(final_logits)

        return output, combined_features