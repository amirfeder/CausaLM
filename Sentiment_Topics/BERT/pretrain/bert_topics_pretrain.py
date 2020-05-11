from transformers.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertModel
from BERT.pretrain.grad_reverse_layer import GradReverseLayerFunction
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch


class BertTopicTreatPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.transform = BertPredictionHeadTransform(config)
        self.pooler = BertTopicTreatPredictionHead.masked_avg_pooler
        # self.pooler = HAN_Attention_Pooler_Layer(config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, 2)
        # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        # self.alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.alpha = 1.

    def forward(self, sequence_output, sequence_mask):
        # hidden_states = self.transform(hidden_states)
        reversed_sequence_output = GradReverseLayerFunction.apply(sequence_output, self.alpha)
        # pooler_seq_mask = self.pooler.create_mask(sequence_mask.sum(dim=-1), sequence_mask.size(-1))
        pooled_output = self.pooler(reversed_sequence_output, sequence_mask)
        # pooled_output, attention_weights = self.pooler(reversed_sequence_output, sequence_mask)
        output = self.decoder(pooled_output)
        return output

    @staticmethod
    def masked_avg_pooler(sequences: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        if masks is None:
            return sequences.mean(dim=1)
        masked_sequences = sequences * masks.float().unsqueeze(dim=-1).expand_as(sequences)
        sequence_lengths = masks.sum(dim=-1).view(-1, 1, 1).expand_as(sequences)
        return torch.sum(masked_sequences / sequence_lengths, dim=1)


class BertTopicControlPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler = BertTopicTreatPredictionHead.masked_avg_pooler
        # self.pooler = HAN_Attention_Pooler_Layer(config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, sequence_mask):
        # pooler_seq_mask = self.pooler.create_mask(sequence_mask.sum(dim=-1), sequence_mask.size(-1))
        pooled_output = self.pooler(sequence_output, sequence_mask)
        # pooled_output, attention_weights = self.pooler(reversed_sequence_output, sequence_mask)
        output = self.decoder(pooled_output)
        return output

    @staticmethod
    def masked_avg_pooler(sequences: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        if masks is None:
            return sequences.mean(dim=1)
        masked_sequences = sequences * masks.float().unsqueeze(dim=-1).expand_as(sequences)
        sequence_lengths = masks.sum(dim=-1).view(-1, 1, 1).expand_as(sequences)
        return torch.sum(masked_sequences / sequence_lengths, dim=1)


class BertTopicTreatPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.topic_treat_prediction = BertTopicTreatPredictionHead(config)

    def forward(self, sequence_output, sequence_mask):
        lm_prediction_scores = self.predictions(sequence_output)
        topic_treat_prediction_score = self.topic_treat_prediction(sequence_output, sequence_mask)
        return lm_prediction_scores, topic_treat_prediction_score


class BertForTopicTreatPreTraining(BertPreTrainedModel):
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
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertTopicTreatPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, masked_lm_labels=None,
                topic_treat_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        lm_prediction_scores, topic_treat_prediction_score = self.cls(sequence_output, attention_mask)

        outputs = (lm_prediction_scores, topic_treat_prediction_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and topic_treat_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            topic_treat_loss = loss_fct(topic_treat_prediction_score.view(-1, 2), topic_treat_label.view(-1))
            loss = masked_lm_loss + topic_treat_loss
            loss_fct_per_sample = CrossEntropyLoss(ignore_index=-1, reduction='none')
            outputs = (loss,
                       torch.stack([loss_fct_per_sample(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                                   .view_as(masked_lm_labels)[i,:].masked_select(masked_lm_labels[i,:] > -1).mean() for i in range(masked_lm_labels.size(0))]),
                       loss_fct_per_sample(topic_treat_prediction_score, topic_treat_label),) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

class BertTopicTreatControlPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.topic_treat_prediction = BertTopicTreatPredictionHead(config)
        self.topic_control_prediction = BertTopicControlPredictionHead(config)

    def forward(self, sequence_output, sequence_mask):
        lm_prediction_scores = self.predictions(sequence_output)
        topic_treat_prediction_score = self.topic_treat_prediction(sequence_output, sequence_mask)
        topic_control_prediction_score = self.topic_control_prediction(sequence_output, sequence_mask)
        return lm_prediction_scores, topic_treat_prediction_score, topic_control_prediction_score


class BertForTopicTreatControlPreTraining(BertPreTrainedModel):
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
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertTopicTreatControlPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, masked_lm_labels=None,
                topic_treat_label=None, topic_control_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        lm_prediction_scores, topic_treat_prediction_score, topic_control_prediction_score = self.cls(sequence_output, attention_mask)

        outputs = (lm_prediction_scores, topic_treat_prediction_score, topic_treat_prediction_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and topic_treat_label is not None and topic_control_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            topic_treat_loss = loss_fct(topic_treat_prediction_score.view(-1, 2), topic_treat_label.view(-1))
            topic_control_loss = loss_fct(topic_control_prediction_score.view(-1, 2), topic_control_label.view(-1))
            loss = masked_lm_loss + topic_treat_loss + topic_control_loss
            loss_fct_per_sample = CrossEntropyLoss(ignore_index=-1, reduction='none')
            outputs = (loss,
                       torch.stack([loss_fct_per_sample(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                                   .view_as(masked_lm_labels)[i,:].masked_select(masked_lm_labels[i,:] > -1).mean() for i in range(masked_lm_labels.size(0))]),
                       loss_fct_per_sample(topic_treat_prediction_score, topic_treat_label),
                       loss_fct_per_sample(topic_control_prediction_score, topic_control_label),) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)
