from torch.nn import Module, Linear
from transformers import BertModel


class Model(Module):
    def __init__(self, num_labels, pretrained_model_name, **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name, output_attentions=False,
                                              output_hidden_states=False)
        bert_output_size = list(self.bert.parameters())[-1].shape[0]
        self.out = Linear(bert_output_size, num_labels)

    def forward(self, desc, attention_mask, **kwargs):
        x = self.bert(desc, attention_mask=attention_mask).pooler_output
        return self.out(x)
