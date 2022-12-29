#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput

class PartModel(nn.Module):
  def __init__(self,num_labels): 
    super(PartModel,self).__init__() 
    self.num_labels = num_labels 
    self.l1=nn.Linear(768, 768)
    self.l2=nn.Linear(768, 768)
    self.softmax = nn.LogSoftmax(dim = 1) 
    self.dropout = nn.Dropout(0.2) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights

  def forward(self, encoded):
    out=self.l1(encoded)
    out=self.dropout(out)
    out2=self.l2(out)
    out2=self.dropout(out2)
    # print(out.shape)
    logits = self.classifier(out2[:,:].view(-1,768)) # calculate losses
    x = self.softmax(logits)
    return x

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'type2.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
