from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertAttention
from constants import BERT_PRETRAINED_MODEL
import torch.nn.functional as F
import torch.nn as nn
import torch


class Linear_Layer(nn.Module):
    def __init__(self, input_size, output_size, dropout=None, batch_norm=False, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if type(dropout) is float and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_size)
        else:
            self.batch_norm = None
        self.activation = activation

    def forward(self, x):
        linear_out = self.linear(x)
        if self.dropout is not None:
            linear_out = self.dropout(linear_out)
        if self.batch_norm is not None:
            linear_out = self.batch_norm(linear_out)
        if self.activation:
            linear_out = self.activation(linear_out)
        return linear_out


class HAN_Attention_Layer(nn.Module):
    def __init__(self, device, h_dim):
        super().__init__()
        self.device = device
        self.linear_in = Linear_Layer(h_dim, h_dim, F.tanh)
        self.softmax = nn.Softmax(dim=-1)
        self.decoder_h = torch.randn(h_dim, device=self.device, requires_grad=True)
        self.weights = dict()

    def forward(self, encoder_h_seq: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            encoder_h_seq (:class:`torch.FloatTensor` [batch size, sequence length, dimensions]): Data
                over which to apply the attention mechanism.
            mask (:class:`torch.ByteTensor` [batch size, sequence length]): Mask
                for padded sequences of variable length.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, seq_len, h_dim = encoder_h_seq.size()

        encoder_h_seq = self.linear_in(encoder_h_seq.contiguous().view(-1, h_dim))
        encoder_h_seq = encoder_h_seq.view(batch_size, seq_len, h_dim)

        # (batch_size, 1, dimensions) * (batch_size, seq_len, dimensions) -> (batch_size, seq_len)
        attention_scores = torch.bmm(self.decoder_h.expand((batch_size, h_dim)).unsqueeze(1), encoder_h_seq.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size, -1)
        if mask is not None:
            attention_scores[~mask] = float("-inf")
        attention_weights = self.softmax(attention_scores)

        # (batch_size, 1, query_len) * (batch_size, query_len, dimensions) -> (batch_size, dimensions)
        output = torch.bmm(attention_weights.unsqueeze(1), encoder_h_seq).squeeze()
        return output, attention_weights

    def create_mask(self, valid_lengths):
        max_len = valid_lengths.max()
        return torch.arange(max_len, dtype=valid_lengths.dtype, device=self.device).expand(len(valid_lengths), max_len) < valid_lengths.unsqueeze(1)


class BertPretrainedClassifier:
    def __init__(self, label_size, batch_size, dropout, device, loss_func=F.cross_entropy,
                 bert_pretrained_model=BERT_PRETRAINED_MODEL, bert_state_dict=None, name="OOB"):
        super().__init__()
        self.name = f"{self.__class__.__name__}-{name}"
        self.device = device
        self.batch_size = batch_size
        self.label_size = label_size
        self.dropout = dropout
        self.loss_func = loss_func
        self.bert = BertPretrainedClassifier.load_frozen_bert(bert_pretrained_model, bert_state_dict)
        # self.config = BertConfigTuple(hidden_size=encoding_dim, num_attention_heads=4,
        #                               attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
        # self.attention = BertAttention(self.bert_config)
        self.pooler = HAN_Attention_Layer(device, self.bert.config.hidden_size)
        self.classifier = Linear_Layer(self.bert.config.hidden_size, label_size, dropout, activation=False)

    def forward(self, input_ids, input_mask, labels):
        last_hidden_states_seq, _ = self.bert(input_ids, attention_mask=input_mask)
        pooled_seq_vector = self.pooler(last_hidden_states_seq)
        logits = self.classifier(pooled_seq_vector)
        loss = self.loss_func(logits.view(-1, self.label_size), labels.view(-1))
        return loss, logits

    @staticmethod
    def load_frozen_bert(bert_pretrained_model, bert_state_dict=None):
        if bert_state_dict:
            fine_tuned_state_dict = torch.load(bert_state_dict)
            bert = BertModel.from_pretrained(bert_pretrained_model, state_dict=fine_tuned_state_dict)
        else:
            bert = BertModel.from_pretrained(bert_pretrained_model)
        for p in bert.parameters():
            p.requires_grad = False
        return bert

    def get_trainable_params(self):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        return parameters, num_trainable_parameters
