from transformers.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertModel
from BERT.lm_finetuning.grad_reverse_layer import GradReverseLayerFunction
from BERT.bert_text_dataset import BertTextDataset
from BERT.bert_pos_tagger import BertTokenClassificationDataset
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch


class BertIMAPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertIMAPredictionHead, self).__init__()
        # self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 2)
        # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        # self.alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.alpha = 1.

    def forward(self, hidden_states):
        # hidden_states = self.transform(hidden_states)
        reversed_hidden_states = GradReverseLayerFunction.apply(hidden_states, self.alpha)
        output = self.decoder(reversed_hidden_states)
        return output


class BertIMAPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertIMAPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.adj_predictions = BertIMAPredictionHead(config)

    def forward(self, sequence_output, pooled_output):
        lm_prediction_scores = self.predictions(sequence_output)
        adj_prediction_scores = self.adj_predictions(sequence_output)
        return lm_prediction_scores, adj_prediction_scores


class BertForIMAPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **masked_adj_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked adjective prediction (classification) loss.
            Indices should be in ``[0, 1]``.
            ``0`` indicates masked word is not adjective,
            ``1`` indicates masked word is adjective.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **adj_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the masked adjective predictions (classification) head (scores of True/False before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForIMAPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertIMAPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, masked_lm_labels=None, masked_adj_labels=None, pos_tagging_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]

        lm_prediction_scores, adj_prediction_scores = self.cls(sequence_output, pooled_output)
        outputs = (lm_prediction_scores, adj_prediction_scores,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and masked_adj_labels is not None:
            loss_f = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX)
            masked_lm_loss = loss_f(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            masked_adj_loss = loss_f(adj_prediction_scores.view(-1, 2), masked_adj_labels.view(-1))
            total_loss = masked_lm_loss + masked_adj_loss
            loss_f_per_sample = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX, reduction='none')
            mlm_loss_per_sample = self.calc_loss_per_sample(loss_f_per_sample, lm_prediction_scores, masked_lm_labels, self.config.vocab_size)
            ima_loss_per_sample = self.calc_loss_per_sample(loss_f_per_sample, adj_prediction_scores, masked_adj_labels, 2)
            outputs = (mlm_loss_per_sample, ima_loss_per_sample,) + outputs

        outputs = (total_loss,) + outputs
        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

    @staticmethod
    def calc_loss_per_sample(loss_f, scores, masked_labels, label_size, ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX):
        return torch.stack([loss_f(scores.view(-1, label_size), masked_labels.view(-1))
                           .view_as(masked_labels)[i, :].masked_select(masked_labels[i, :] > ignore_index).mean()
                            for i in range(masked_labels.size(0))])


class BertTokenClassificationHead(nn.Module):
    def __init__(self, config):
        super(BertTokenClassificationHead, self).__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states):
        output = self.classifier(hidden_states)
        return output


class BertIMAwControlPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertIMAwControlPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.adj_predictions = BertIMAPredictionHead(config)
        self.pos_tagging = BertTokenClassificationHead(config)

    def forward(self, sequence_output, pooled_output):
        lm_prediction_scores = self.predictions(sequence_output)
        adj_prediction_scores = self.adj_predictions(sequence_output)
        pos_tagging_scores = self.pos_tagging(sequence_output)
        return lm_prediction_scores, adj_prediction_scores, pos_tagging_scores


class BertForIMAwControlPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **masked_adj_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked adjective prediction (classification) loss.
            Indices should be in ``[0, 1]``.
            ``0`` indicates masked word is not adjective,
            ``1`` indicates masked word is adjective.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **adj_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the masked adjective predictions (classification) head (scores of True/False before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForIMAwControlPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertIMAwControlPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, masked_lm_labels=None, masked_adj_labels=None, pos_tagging_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]

        lm_prediction_scores, adj_prediction_scores, pos_tagging_scores = self.cls(sequence_output, pooled_output)
        outputs = (lm_prediction_scores, adj_prediction_scores, pos_tagging_scores,) + outputs[2:]  # add hidden states and attention if they are here

        total_loss = 0.0

        if pos_tagging_labels is not None:
            loss_f = CrossEntropyLoss(ignore_index=BertTokenClassificationDataset.POS_IGNORE_LABEL_IDX)
            loss_f_per_sample = CrossEntropyLoss(ignore_index=BertTokenClassificationDataset.POS_IGNORE_LABEL_IDX, reduction='none')
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = pos_tagging_scores.view(-1, self.config.num_labels)
                active_labels = torch.where(
                    active_loss, pos_tagging_labels.view(-1), torch.tensor(loss_f.ignore_index).type_as(pos_tagging_labels)
                )
                pos_tagging_loss = loss_f(active_logits, active_labels)
                # pos_tagging_loss_per_sample = BertForIMAPreTraining.calc_loss_per_sample(loss_f_per_sample,
                #                                                                          active_logits,
                #                                                                          active_labels,
                #                                                                          self.config.num_labels)
            else:
                pos_tagging_loss = loss_f(pos_tagging_scores.view(-1, self.config.num_labels), pos_tagging_labels.view(-1))

            pos_tagging_loss_per_sample = BertForIMAPreTraining.calc_loss_per_sample(loss_f_per_sample,
                                                                                     pos_tagging_scores,
                                                                                     pos_tagging_labels,
                                                                                     self.config.num_labels,
                                                                                     BertTokenClassificationDataset.POS_IGNORE_LABEL_IDX)
            total_loss += pos_tagging_loss
            outputs = (pos_tagging_loss_per_sample,) + outputs

        if masked_lm_labels is not None and masked_adj_labels is not None:
            loss_f = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX)
            masked_lm_loss = loss_f(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            masked_adj_loss = loss_f(adj_prediction_scores.view(-1, 2), masked_adj_labels.view(-1))
            total_loss += masked_lm_loss + masked_adj_loss
            loss_f_per_sample = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX, reduction='none')
            mlm_loss_per_sample = BertForIMAPreTraining.calc_loss_per_sample(loss_f_per_sample,
                                                                             lm_prediction_scores,
                                                                             masked_lm_labels,
                                                                             self.config.vocab_size)
            ima_loss_per_sample = BertForIMAPreTraining.calc_loss_per_sample(loss_f_per_sample,
                                                                             adj_prediction_scores,
                                                                             masked_adj_labels,
                                                                             2)
            outputs = (mlm_loss_per_sample, ima_loss_per_sample,) + outputs

        outputs = (total_loss,) + outputs
        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)
