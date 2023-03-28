import torch
from torch.nn import Module, Linear, functional
from transformers import BertModel


class Model(Module):
    def __init__(self, num_labels, pretrained_model_name, shop_id_vector_size, cat_hid_size=16):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name, output_attentions=False,
                                              output_hidden_states=False)
        bert_output_size = list(self.bert.parameters())[-1].shape[0]
        self.categorical = Linear(shop_id_vector_size, cat_hid_size)
        self.out = Linear(bert_output_size + cat_hid_size, num_labels)

    def forward(self, desc, attention_mask, shop_id, **kwargs):
        shop_id = shop_id.squeeze(1)
        x_desc = self.bert(desc, attention_mask=attention_mask).pooler_output
        x_shop = functional.relu(self.categorical(shop_id))
        x = torch.cat((x_desc, x_shop), dim=1)
        return self.out(x)
