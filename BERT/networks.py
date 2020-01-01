from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertAttention
from torch.utils.data.dataloader import DataLoader
from dataset import BertSentimentDataset, BERT_PRETRAINED_MODEL
from pytorch_lightning import LightningModule, data_loader
from utils import save_predictions
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

    def create_mask(self, valid_lengths, max_len=None):
        if not max_len:
            max_len = valid_lengths.max()
        return torch.arange(max_len, dtype=valid_lengths.dtype, device=self.device).expand(len(valid_lengths), max_len) < valid_lengths.unsqueeze(1)


class BertPretrainedClassifier(nn.Module):
    def __init__(self, device, batch_size, dropout, label_size=2, loss_func=F.cross_entropy,
                 bert_pretrained_model=BERT_PRETRAINED_MODEL, bert_state_dict=None, name="OOB"):
        super().__init__()
        self.name = f"{self.__class__.__name__}-{name}"
        self.device = device
        self.batch_size = batch_size
        self.label_size = label_size
        self.dropout = dropout
        self.loss_func = loss_func
        self.bert_pretrained_model = bert_pretrained_model
        self.bert_state_dict = bert_state_dict
        self.bert = BertPretrainedClassifier.load_frozen_bert(bert_pretrained_model, bert_state_dict)
        # self.config = BertConfigTuple(hidden_size=encoding_dim, num_attention_heads=4,
        #                               attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
        # self.attention = BertAttention(self.bert_config)
        self.hidden_size = self.bert.config.hidden_size
        self.pooler = HAN_Attention_Layer(device, self.hidden_size)
        self.classifier = Linear_Layer(self.hidden_size, label_size, dropout, activation=False)

    def forward(self, input_ids, input_mask, labels):
        last_hidden_states_seq, _ = self.bert(input_ids, attention_mask=input_mask)
        pooler_mask = self.pooler.create_mask(input_mask.sum(dim=1), input_mask.size(1))
        pooled_seq_vector, attention_weights = self.pooler(last_hidden_states_seq, pooler_mask)
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

    def save_model(self, kwargs=None, path=None, filename=None):
        model_dict = {'name': self.name,
                      'device': self.device,
                      'batch_size': self.batch_size,
                      'label_size': self.label_size,
                      'dropout': self.dropout,
                      'loss_func': self.loss_func,
                      'state_dict': self.state_dict()
                      }
        model_save_name = self.name
        if kwargs:
            model_dict['external'] = kwargs
        if filename:
            model_save_name = f"{model_save_name}_{filename}"
        torch.save(model_dict, f"{path}/{model_save_name}.pt")


class Hyperparameters:
    def __init__(self, kwargs):
        self.__dict__ = kwargs


class LightningBertPretrainedClassifier(LightningModule):
    def __init__(self, output_path, data_path, treatment, text_column, label_column, **bert_params):
        super().__init__()
        self.output_path = output_path
        self.data_path = data_path
        self.treatment = treatment
        self.text_column = text_column
        self.label_column = label_column
        self.bert_classifier = BertPretrainedClassifier(**bert_params)
        self.hparams = Hyperparameters({"data_path": data_path,
                        "treatment": treatment,
                        "text_column": text_column,
                        "label_column": label_column,
                        "model_name": self.bert_classifier.name,
                        "device": self.bert_classifier.device,
                        "batch_size": self.bert_classifier.batch_size,
                        "dropout": self.bert_classifier.dropout,
                        "label_size": self.bert_classifier.label_size,
                        "loss_func": self.bert_classifier.loss_func,
                        "pretrained_model": self.bert_classifier.bert_pretrained_model,
                        "bert_state_dict": self.bert_classifier.bert_state_dict})

    def configure_optimizers(self):
        return torch.optim.Adam(self.bert_classifier.get_trainable_params()[0])

    def forward(self, *args):
        return self.bert_classifier.forward(*args)

    @data_loader
    def train_dataloader(self):
        dataset = BertSentimentDataset(self.data_path, self.treatment, "train", self.text_column, self.label_column)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True)
        return dataloader

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        loss, logits = self.forward(input_ids, input_mask, labels)
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        total = torch.tensor(predictions.size(0))
        return {"loss": loss, "progress_bar": {"train_loss": loss, "train_accuracy": correct.mean()},
                "log": {"train_loss": loss, "train_accuracy": correct.mean()}, "correct": correct.sum(), "total": total}

    def training_end(self, step_outputs):
        loss = step_outputs["loss"]
        accuracy = step_outputs["correct"] / float(step_outputs["total"])
        return {"loss": loss, "progress_bar": {"train_loss": loss, "train_accuracy": accuracy},
                "log": {"train_loss": loss, "train_accuracy": accuracy}}

    @data_loader
    def val_dataloader(self):
        dataset = BertSentimentDataset(self.data_path, self.treatment, "dev", self.text_column, self.label_column)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True)
        return dataloader

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        loss, logits = self.forward(input_ids, input_mask, labels)
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        return {"val_loss": loss, "progress_bar": {"val_loss": loss, "val_accuracy": correct.mean()},
                "log": {"val_loss": loss, "val_accuracy": correct.mean()}, "correct": correct}

    def validation_end(self, step_outputs):
        total_loss, total_correct = list(), list()
        for x in step_outputs:
            total_loss.append(x["val_loss"])
            total_correct.append(x["correct"])
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.stack(total_correct).double()
        return {"progress_bar": {"val_loss": avg_loss, "val_accuracy": accuracy.mean()},
                "log": {"avg_val_loss": avg_loss,
                        "avg_val_accuracy": accuracy.mean(),
                        "max_val_accuracy": accuracy.max(),
                        "min_val_accuracy": accuracy.min()}}

    @data_loader
    def test_dataloader(self):
        dataset = BertSentimentDataset(self.data_path, self.treatment, "test", self.text_column, self.label_column)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True)
        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        loss, logits = self.forward(input_ids, input_mask, labels)
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        return {"test_loss": loss, "progress_bar": {"test_loss": loss, "test_accuracy": correct.mean()},
                "log": {"test_loss": loss, "test_accuracy": correct.mean()},
                "predictions": predictions, "labels": labels, "unique_ids": unique_ids}

    def test_end(self, step_outputs):
        total_loss, total_predictions, total_labels, total_unique_ids = list(), list(), list(), list()
        for x in step_outputs:
            total_loss.append(x["test_loss"])
            total_predictions.append(x["predictions"])
            total_labels.append(x["labels"])
            total_unique_ids.append(x["unique_ids"])
        avg_loss = torch.stack(total_loss).mean()
        unique_ids = torch.cat(total_unique_ids)
        predictions = torch.cat(total_predictions)
        labels = torch.cat(total_labels)
        accuracy = predictions.eq(labels.view_as(predictions)).double()
        save_predictions(self.output_path, unique_ids.cpu().numpy(), predictions.cpu().numpy(), labels.cpu().numpy(), "test")
        return {"progress_bar": {"test_loss": avg_loss, "test_accuracy": accuracy.mean()},
                "log": {"avg_test_accuracy": accuracy.mean(),
                        "max_test_accuracy": accuracy.max(),
                        "max_test_accuracy_epoch": accuracy.argmax() + 1,
                        "min_test_accuracy": accuracy.min()}}


