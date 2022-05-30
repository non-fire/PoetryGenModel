import torch
from torch import nn
import torch.nn.functional as F

class PoetryGenModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layer_num):
        super(PoetryGenModel, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.layer_num=layer_num

        self.Embedding=nn.Embedding(vocab_size,embedding_size)
        self.LSTM=nn.LSTM(embedding_size,hidden_size,num_layers=layer_num)
        self.Linear=nn.Linear(hidden_size,vocab_size)
        #self.Flatten=nn.Flatten()
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        #print("input",input.size())
        batch_size, num_steps = input.size()[1],input.size()[0]
        embeds=self.Embedding(input)
        #print("embeds",embeds.size())
        Y,hidden=self.LSTM(embeds,hidden)
        #print("LSTM",Y.size())
        #Y=self.Linear(self.Flatten(F.relu(Y.reshape((batch_size*num_steps,-1)))))
        Y=Y.reshape((batch_size * num_steps, -1))
        #print("reshape",Y.size())
        Y=self.Linear(Y)
        Y = F.relu(Y)
        #print("Linear",Y.size())
        output=self.softmax(Y)
        #print("softmax",output.size())
        return output,hidden

    def begin_state(self, batch_size):
        return (torch.zeros((self.layer_num, batch_size, self.hidden_size)).to(self.device),
                torch.zeros((self.layer_num, batch_size, self.hidden_size)).to(self.device))


