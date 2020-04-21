from typing import Callable, List
from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertAttention
from torch.utils.data.dataloader import DataLoader
from constants import NUM_CPU
from BERT.dataset import BertTextClassificationDataset, BERT_PRETRAINED_MODEL
from pytorch_lightning import LightningModule, data_loader
from utils import save_predictions
import torch.nn.functional as F
import torch.nn as nn
import torch


class Linear_Layer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = None,
                 batch_norm: bool = False, layer_norm: bool = False, activation: Callable = F.relu):
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
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        if self.dropout:
            linear_out = self.dropout(linear_out)
        if self.batch_norm:
            linear_out = self.batch_norm(linear_out)
        if self.layer_norm:
            linear_out = self.layer_norm(linear_out)
        if self.activation:
            linear_out = self.activation(linear_out)
        return linear_out


class HAN_Attention_Pooler_Layer(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.linear_in = Linear_Layer(h_dim, h_dim, activation=torch.tanh)
        self.softmax = nn.Softmax(dim=-1)
        self.decoder_h = nn.Parameter(torch.randn(h_dim), requires_grad=True)

    def forward(self, encoder_h_seq: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            encoder_h_seq (:class:`torch.FloatTensor` [batch size, sequence length, dimensions]): Data
                over which to apply the attention mechanism.
            mask (:class:`torch.BoolTensor` [batch size, sequence length]): Mask
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
            if mask.dtype is not torch.bool:
                mask = mask.bool()
            attention_scores[~mask] = float("-inf")
        attention_weights = self.softmax(attention_scores)

        # (batch_size, 1, query_len) * (batch_size, query_len, dimensions) -> (batch_size, dimensions)
        output = torch.bmm(attention_weights.unsqueeze(1), encoder_h_seq).squeeze()
        return output, attention_weights

    @staticmethod
    def create_mask(valid_lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
        if not max_len:
            max_len = valid_lengths.max()
        return torch.arange(max_len, dtype=valid_lengths.dtype, device=valid_lengths.device).expand(len(valid_lengths), max_len) < valid_lengths.unsqueeze(1)


class BertPretrainedClassifier(nn.Module):
    def __init__(self, batch_size: int = 8, dropout: float = 0.1, label_size: int = 2,
                 loss_func: Callable = F.cross_entropy, bert_pretrained_model: str = BERT_PRETRAINED_MODEL,
                 bert_state_dict: str = None, name: str = "OOB", device: torch.device = None):
        super().__init__()
        self.name = f"{self.__class__.__name__}-{name}"
        self.batch_size = batch_size
        self.label_size = label_size
        self.dropout = dropout
        self.loss_func = loss_func
        self.device = device
        self.bert_pretrained_model = bert_pretrained_model
        self.bert_state_dict = bert_state_dict
        self.bert = BertPretrainedClassifier.load_frozen_bert(bert_pretrained_model, bert_state_dict)
        # self.config = BertConfigTuple(hidden_size=encoding_dim, num_attention_heads=4,
        #                               attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
        # self.attention = BertAttention(self.bert_config)
        self.hidden_size = self.bert.config.hidden_size
        self.pooler = HAN_Attention_Pooler_Layer(self.hidden_size)
        self.classifier = Linear_Layer(self.hidden_size, label_size, dropout, activation=None)

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        last_hidden_states_seq, _ = self.bert(input_ids, attention_mask=input_mask)
        # pooler_mask = self.pooler.create_mask(input_mask.sum(dim=1), input_mask.size(1))
        pooled_seq_vector, attention_weights = self.pooler(last_hidden_states_seq, input_mask)
        logits = self.classifier(pooled_seq_vector)
        loss = self.loss_func(logits.view(-1, self.label_size), labels.view(-1))
        return loss, logits, attention_weights

    @staticmethod
    def load_frozen_bert(bert_pretrained_model: str, bert_state_dict: str = None) -> BertModel:
        if bert_state_dict:
            fine_tuned_state_dict = torch.load(bert_state_dict)
            bert = BertModel.from_pretrained(bert_pretrained_model, state_dict=fine_tuned_state_dict)
        else:
            bert = BertModel.from_pretrained(bert_pretrained_model)
        for p in bert.parameters():
            p.requires_grad = False
        return bert

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        return parameters, num_trainable_parameters

    def save_model(self, kwargs=None, path=None, filename=None):
        model_dict = {'name': self.name,
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


class LightningHyperparameters:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


# LightningHyperparameters = lambda hparams_dict: \
#     namedtuple("LightningHyperparameters", hparams_dict.keys())(**hparams_dict)


class LightningBertPretrainedClassifier(LightningModule):
    def __init__(self, hparams: LightningHyperparameters):
        super().__init__()
        self.hparams = hparams
        self.bert_classifier = BertPretrainedClassifier(**hparams.bert_params)

    # def parameters(self, recurse: bool = True):
    #     for param in self.bert_classifier.get_trainable_params(recurse)[0]:
    #         yield param

    def parameters(self, recurse: bool = ...):
        return self.bert_classifier.parameters(recurse)

    def configure_optimizers(self):
        parameters_list = self.bert_classifier.get_trainable_params()[0]
        if parameters_list:
            return torch.optim.Adam(parameters_list)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    def forward(self, *args):
        return self.bert_classifier.forward(*args)

    @data_loader
    def train_dataloader(self):
        if not self.training:
            return [] # PyTorch Lightning hack for test mode with frozen model
        dataset = BertTextClassificationDataset(self.hparams.data_path, self.hparams.treatment, "train",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        loss, logits, pooler_attention_weights = self.forward(input_ids, input_mask, labels)
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        total = torch.tensor(predictions.size(0))
        return {"loss": loss, "log": {"batch_num": batch_idx, "train_loss": loss, "train_accuracy": correct.mean()},
                "correct": correct.sum(), "total": total}

    # def training_end(self, step_outputs):
    #     loss = step_outputs["loss"]
    #     accuracy = step_outputs["correct"] / float(step_outputs["total"])
    #     return {"loss": loss, "progress_bar": {"train_loss": loss, "train_accuracy": accuracy},
    #             "log": {"train_loss": loss, "train_accuracy": accuracy}}

    @data_loader
    def val_dataloader(self):
        if not self.training:
            return [] # PyTorch Lightning hack for test mode with frozen model
        dataset = BertTextClassificationDataset(self.hparams.data_path, self.hparams.treatment, "dev",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        loss, logits, pooler_attention_weights = self.forward(input_ids, input_mask, labels)
        prediction_probs = F.softmax(logits.view(-1, self.bert_classifier.label_size), dim=-1)
        predictions = torch.argmax(prediction_probs, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        return {"loss": loss, "progress_bar": {"val_loss": loss, "val_accuracy": correct.mean()},
                "log": {"batch_num": batch_idx, "val_loss": loss, "val_accuracy": correct.mean()}, "correct": correct}

    def validation_end(self, step_outputs):
        total_loss, total_correct = list(), list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            total_correct.append(x["correct"])
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.cat(total_correct).mean()
        return {"loss": avg_loss, "progress_bar": {"val_loss": avg_loss, "val_accuracy": accuracy},
                "log": {"val_loss": avg_loss, "val_accuracy": accuracy}}

    @data_loader
    def test_dataloader(self):
        dataset = BertTextClassificationDataset(self.hparams.data_path, self.hparams.treatment, "test",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        loss, logits, pooler_attention_weights = self.forward(input_ids, input_mask, labels)
        prediction_probs = F.softmax(logits.view(-1, self.bert_classifier.label_size), dim=-1)
        predictions = torch.argmax(prediction_probs, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        return {"loss": loss, "progress_bar": {"test_loss": loss, "test_accuracy": correct.mean()},
                "log": {"batch_num": batch_idx, "test_loss": loss, "test_accuracy": correct.mean()},
                "predictions": predictions, "labels": labels, "unique_ids": unique_ids, "prediction_probs": prediction_probs}

    def test_end(self, step_outputs):
        total_loss, total_predictions, total_labels, total_unique_ids, total_prediction_probs = list(), list(), list(), list(), list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            total_predictions.append(x["predictions"])
            total_labels.append(x["labels"])
            total_unique_ids.append(x["unique_ids"])
            total_prediction_probs.append(x["prediction_probs"])
        avg_loss = torch.stack(total_loss).double().mean()
        unique_ids = torch.cat(total_unique_ids).long()
        predictions = torch.cat(total_predictions).long()
        prediction_probs = torch.cat(total_prediction_probs, dim=0).double()
        labels = torch.cat(total_labels).long()
        correct = predictions.eq(labels.view_as(predictions)).long()
        accuracy = correct.double().mean()
        save_predictions(self.hparams.output_path,
                         unique_ids.data.cpu().numpy(),
                         predictions.data.cpu().numpy(),
                         labels.data.cpu().numpy(),
                         correct.cpu().numpy(),
                         [prediction_probs[:, i].data.cpu().numpy() for i in range(self.bert_classifier.label_size)],
                         f"{self.bert_classifier.name}-test")
        return {"loss": avg_loss,
                "progress_bar": {"test_loss": avg_loss, "test_accuracy": accuracy},
                "log": {"test_loss": avg_loss, "test_accuracy": accuracy}}


