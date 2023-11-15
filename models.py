import torch.nn as nn
class BERT_Arch(nn.Module):
    def __init__(self, bert, dropout=0.1):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2) # HAM vs SPAM (2 LABELS)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        outputs = self.bert(sent_id, attention_mask=mask)
#         print(cls_hs)
#         x = self.fc1(outputs.last_hidden_state)
        x = self.fc1(outputs.pooler_output)
        #x dim 512
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x