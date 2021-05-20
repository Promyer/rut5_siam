import torch
from torch import nn

from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import copy


class ML(nn.Module):

    '''
    Multi linear layer with dropout 0.3, batch norm, residual and ReLU activation
    '''
    def __init__():
        pass

    def forward(self, x):
        pass


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class T5Siamese(T5PreTrainedModel):
    def __init__(self, config, head_sizes, classifier_sizes, deep=None):
        super().__init__(config)
        config.num_labels = 1
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = T5Stack(encoder_config, self.shared)
        self.question_head = ML(
            sizes=head_sizes,
            dropout=0.3
        )
        self.response_head = ML(
            sizes=head_sizes,
            dropout=0.3
        )
        self.classifier = ML(
            sizes=classifier_sizes,
            dropout=0.3
        )
        self.init_weights()
        self.deep = deep

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        query_input,
        answer_input,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.deep == "head":
            with torch.no_grad():
                query_outputs = self.encoder(
                    **query_input,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                answer_outputs = self.encoder(
                    **answer_input,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            query_outputs = self.encoder(
                **query_input,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            answer_outputs = self.encoder(
                **answer_input,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        query_pooled_output = mean_pooling(query_outputs, query_input["attention_mask"])
        answer_pooled_output = mean_pooling(answer_outputs, answer_input["attention_mask"])

        query_pooled_output = self.question_head(query_pooled_output)
        answer_pooled_output = self.response_head(answer_pooled_output)
        pooled_output = torch.cat([query_pooled_output, answer_pooled_output], dim=1)
        logits = self.classifier(pooled_output)

        return logits.squeeze(1)
