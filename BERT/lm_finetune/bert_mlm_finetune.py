from BERT.bert_text_dataset import BertTextDataset
from transformers.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn


class BertMLMPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertMLMPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output, pooled_output):
        lm_prediction_scores = self.predictions(sequence_output)
        return lm_prediction_scores


class BertForMLMPreTraining(BertPreTrainedModel):
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
        super(BertForMLMPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertMLMPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, masked_lm_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        lm_prediction_scores = self.cls(sequence_output, pooled_output)

        outputs = (lm_prediction_scores,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX)
            masked_lm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            loss = masked_lm_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)
