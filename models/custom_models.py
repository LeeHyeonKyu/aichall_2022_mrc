from torch import nn
from transformers import ElectraForQuestionAnswering

F = nn.functional

class electra(nn.Module):

    def __init__(self, pretrained, **kwargs):
        super(electra, self).__init__()

        self.model = ElectraForQuestionAnswering.from_pretrained(pretrained)
        

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        
        outputs = self.model(input_ids = input_ids, 
                             attention_mask = attention_mask,
                             start_positions = start_positions,
                             end_positions = end_positions)
        
        return outputs


 