from ast import literal_eval
from typing import List
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningModule, data_loader
from tqdm import tqdm
from transformers import BertConfig, BertForTokenClassification
from BERT.bert_text_dataset import BERT_PRETRAINED_MODEL, BertTextDataset, InputExample, InputFeatures, InputLabel, \
    truncate_seq_first
from BERT.bert_text_classifier import LightningHyperparameters
import torch.nn as nn
import torch.nn.functional as F
import torch

from datasets.utils import POS_TAGS_TUPLE, CLS_TOKEN, SEP_TOKEN, TOKEN_SEPARATOR
from constants import NUM_CPU, MAX_SENTIMENT_SEQ_LENGTH
from utils import save_predictions


class BertPOSTagger(LightningModule):
    #TODO: Verify all lightning train,val,test methods work properly for sequence tagging tasks
    def __init__(self, hparams: LightningHyperparameters):
        super().__init__()
        self.haparams = hparams
        self.bert_pretrained_model = hparams.bert_model if hasattr(hparams, "bert_model") else BERT_PRETRAINED_MODEL
        self.bert_state_dict = hparams.bert_state_dict if hasattr(hparams, "bert_state_dict") else None
        self.num_labels = hparams.num_labels if hasattr(hparams, "num_labels") else len(POS_TAGS_TUPLE)
        self.bert_token_classifier = BertPOSTagger.load_frozen_bert(self.bert_model, self.bert_state_dict,
                                                                    BertConfig.from_pretrained(self.bert_model,
                                                                                               num_labels=self.num_labels))

    @staticmethod
    def load_frozen_bert(bert_pretrained_model: str, bert_state_dict: str = None, bert_config: BertConfig = None) -> BertForTokenClassification:
        if bert_state_dict:
            fine_tuned_state_dict = torch.load(bert_state_dict)
            bert_token_classifier = BertForTokenClassification.from_pretrained(pretrained_model_name_or_path=bert_pretrained_model,
                                                                               state_dict=fine_tuned_state_dict,
                                                                               config=bert_config)
        else:
            bert_token_classifier = BertForTokenClassification.from_pretrained(pretrained_model_name_or_path=bert_pretrained_model,
                                                                               config=bert_config)
        for p in bert_token_classifier.bert.parameters():
            p.requires_grad = False
        return bert_token_classifier

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        return self.bert_token_classifier.forward(input_ids=input_ids, attention_mask=attention_mask,
                                                  token_type_ids=token_type_ids, position_ids=position_ids,
                                                  head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels)

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        return parameters, num_trainable_parameters

    def parameters(self, recurse: bool = ...):
        return self.bert_token_classifier.parameters(recurse)

    def configure_optimizers(self):
        parameters_list = self.bert_token_classifier.get_trainable_params()[0]
        if parameters_list:
            return torch.optim.Adam(parameters_list)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    @data_loader
    def train_dataloader(self):
        if not self.training:
            return [] # PyTorch Lightning hack for test mode with frozen model
        dataset = BertTokenClassificationDataset(self.hparams.data_path, self.hparams.treatment, "train",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        net_out = self.forward(input_ids=input_ids, attention_mask=input_mask, labels=labels)
        loss, logits = net_out[:2]
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        total = torch.tensor(predictions.size(0))
        return {"loss": loss, "log": {"batch_num": batch_idx, "train_loss": loss, "train_accuracy": correct.mean()},
                "correct": correct.sum(), "total": total}

    @data_loader
    def val_dataloader(self):
        if not self.training:
            return [] # PyTorch Lightning hack for test mode with frozen model
        dataset = BertTokenClassificationDataset(self.hparams.data_path, self.hparams.treatment, "dev",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        net_out = self.forward(input_ids, input_mask, labels)
        loss, logits = net_out[:2]
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
        dataset = BertTokenClassificationDataset(self.hparams.data_path, self.hparams.treatment, "test",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.bert_classifier.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids = batch
        net_out = self.forward(input_ids, input_mask, labels)
        loss, logits = net_out[:2]
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


class BertTokenClassificationDataset(BertTextDataset):

    POS_IGNORE_LABEL_IDX = -100

    def __init__(self, data_path: str, treatment: str, subset: str, text_column: str, label_column: str,
                 bert_pretrained_model: str = BERT_PRETRAINED_MODEL,
                 max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH):
        super().__init__(data_path, treatment, subset, text_column, label_column, bert_pretrained_model,
                         max_seq_length)

    def read_examples_func(self, row):
        unique_id = int(row.iloc[0])
        text = str(row[self.text_column])
        labels_list = [int(i) for i in literal_eval(str(row[self.label_column]))]
        return InputExample(unique_id=unique_id, text=text, label=tuple(labels_list))

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputFeature`s."""
        features_list = list()
        labels_list = list()
        # seq_lengths = list()
        for i, example in tqdm(enumerate(examples), total=len(examples),
                               desc=f"{self.subset}-convert_examples_to_features"):
            features, example_len, labels = self.tokenize_and_pad_sequence(example)
            features_list.append(features)
            labels_list.append(labels)
            # seq_lengths.append(example_len)
        # print_seq_lengths_stats(None, seq_lengths, self.max_seq_length)
        return features_list, labels_list

    def tokenize_and_pad_sequence(self, example):
        tokens = self.tokenizer.tokenize(example.text)
        labels = self.align_labels_to_bert_tokenization(self.tokenizer, tokens, example.text, list(example.label))

        tokens = tuple([CLS_TOKEN] + truncate_seq_first(tokens, self.max_seq_length) + [SEP_TOKEN])
        labels = [self.POS_IGNORE_LABEL_IDX] + truncate_seq_first(labels, self.max_seq_length) + [
            self.POS_IGNORE_LABEL_IDX]

        example_len = len(tokens) - 2

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.PAD_TOKEN_IDX)
            input_mask.append(self.PAD_TOKEN_IDX)
            labels.append(self.POS_IGNORE_LABEL_IDX)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(labels) == self.max_seq_length

        return InputFeatures(unique_id=example.unique_id, tokens=tokens,
                             input_ids=tuple(input_ids), input_mask=tuple(input_mask)), \
               example_len, \
               InputLabel(unique_id=example.unique_id, label=tuple(labels))

    @staticmethod
    def align_labels_to_bert_tokenization(bert_tokenizer, tokenized_text, orig_text, orig_labels):
        # Align pos tag labels according to BERT tokenization
        if type(orig_text) is list:
            orig_text_tokens = orig_text
        elif type(orig_text) is str:
            orig_text_tokens = orig_text.split(TOKEN_SEPARATOR)

        tokenizer_adds = [len(bert_tokenizer.tokenize(token)) - 1 for token in orig_text_tokens]
        aligned_labels = list(orig_labels)
        assert len(tokenizer_adds) == len(orig_labels)

        num_adds = 0
        num_removes = 0
        for idx, i in enumerate(tokenizer_adds):
            for j in range(abs(i)):
                label_idx = idx + num_adds - num_removes
                if i > 0:
                    aligned_labels.insert(label_idx, aligned_labels[label_idx - 1])
                    num_adds += 1
                elif i < 0:
                    aligned_labels.pop(label_idx)
                    num_removes += 1

        assert num_adds - num_removes == sum(tokenizer_adds), f"{num_adds} {sum(tokenizer_adds)}"
        assert len(tokenized_text) == len(aligned_labels), f"\n{tokenized_text}\n{aligned_labels}\n{orig_labels}"

        return aligned_labels