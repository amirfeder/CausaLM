import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2, add_FC_layer=0):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.add_FC_layer = add_FC_layer
        if add_FC_layer:
            self.fc_layer = nn.Linear(config.hidden_size, 128)
            self.classifier = nn.Linear(128, num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        if self.add_FC_layer:
            pooled_output = torch.sigmoid(self.fc_layer(pooled_output))
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



class CNNBertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification and masked Pivot feature prediction.
    This module is composed of the BERT model with two linear layer on top of
    the pooled output - one for classification task and the other for multi-label classification task.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
        `num_aux_labels`: the number of classes for the auxiliary task classifier. Default = 500.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token.
             (see the tokens preprocessing logic in the scripts `extract_features.py`, `run_classifier.py` and
              `run_squad.py`). When training for the auxiliary task the input pivot features are masked.
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `multy_class_labels`: multy-class labels for the auxiliary task classification output:
        torch.LongTensor of shape [batch_size, num_of_pivots] with indices selected in [0, 1].
    Outputs:
        if `labels` is not `None` and multy_class_labels is not `None`:
            Outputs the CrossEntropy classification loss for labeled data + CrossEntropy Multi Class Binary
            classification loss for unlabeled data.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, -1, 1], [-1, 1, 0]]) # '-1' for unlabeled data
    token_aux_ids = torch.LongTensor([[0, 0, ..., 1, ..., 0, 1, ... , 0], [0, 1, ..., 0, ..., 0, 0, ... , 1]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    num_aux_labels = 500
    model = BertForSequenceClassificationWithAux(config, num_labels, num_aux_labels)
    logits = model(input_ids, token_type_ids, token_aux_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2, hidden_size=768, filter_size=9, out_channels=16, max_seq_length=128,
                 padding=True, output_layer_num=12):
        super(CNNBertForSequenceClassification, self).__init__(config)
        self.output_layer_num = output_layer_num
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        padding_size = int((filter_size-1)/2) if padding else 0
        self.conv1 = nn.Conv1d(in_channels=hidden_size,
                               out_channels=out_channels,
                               kernel_size=filter_size, padding=padding_size)

        # self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.max_pool = nn.AvgPool1d(kernel_size=2)
        classifier_in_size = int(out_channels*max_seq_length/2) if padding else \
            int((out_channels*(max_seq_length-filter_size+1))/2)
        self.classifier = nn.Linear(classifier_in_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, aux_multi_labels=None):
        if self.output_layer_num == 12:
            enc_sequence, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        else:
            enc_sequence, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
            enc_sequence = enc_sequence[self.output_layer_num-1]
        # Mask bert output where input was 0
        # enc_sequence = torch.mul(enc_sequence, (
        #     torch.tensor(attention_mask, dtype=torch.float, device=torch.device('cuda')).unsqueeze(2).repeat(1, 1, int(
        #         enc_sequence.shape[2]))))
        enc_sequence = self.dropout(enc_sequence)
        enc_seq_shape = enc_sequence.shape
        enc_sequence = enc_sequence.reshape(enc_seq_shape[0], enc_seq_shape[2], enc_seq_shape[1])
        features = self.conv1(enc_sequence)
        features_shape = features.shape
        features = features.reshape(features_shape[0], features_shape[2], features_shape[1])
        final_features_shape = features.shape
        final_features = features.reshape(final_features_shape[0], final_features_shape[2], final_features_shape[1])
        final_features = self.max_pool(final_features)
        final_features_shape = final_features.shape
        flat = final_features.reshape(-1, final_features_shape[1]*final_features_shape[2])
        logits = self.classifier(flat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
