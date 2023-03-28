import torch
from torch.nn import Module, Linear, functional
from torchvision.models import resnet18
from transformers import BertModel


class Model(Module):
    def __init__(self, num_labels, pretrained_model_name, **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name, output_attentions=False,
                                              output_hidden_states=False)
        bert_output_size = list(self.bert.parameters())[-1].shape[0]
        self.resnet = resnet18(pretrained=True)
        resnet_output_size = list(self.resnet.parameters())[-1].shape[-1]
        self.out = Linear(bert_output_size + resnet_output_size, num_labels)

    def forward(self, desc, attention_mask, img, **kwargs):
        x_desc = self.bert(desc, attention_mask=attention_mask).pooler_output
        x_img = self.resnet(img)
        x = torch.cat((x_desc, x_img), dim=1)
        return self.out(x)
