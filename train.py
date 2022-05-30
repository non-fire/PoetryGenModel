import torch
from torch import nn
from torch.nn import functional as F
import dataLoader
import torch.utils.data as Data
#import sample
from model import PoetryGenModel

batch_size, num_steps = 256, 30
layer_num=2
#weight_decay=1e-4
lr=0.01
num_epochs=30
data, vocab = dataLoader.dataHandler(batch_size, num_steps)
train_iter = Data.DataLoader(data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
embedding_size=hidden_size=256

model=PoetryGenModel(len(vocab), embedding_size, hidden_size, layer_num)
model=model.to(model.device)

def train_epoch(net, train_iter, loss, updater):
    state=None
    cnt = 0
    loss_sum = 0
    for x in train_iter:
        state = net.begin_state(batch_size)
        x = x.long().transpose(1, 0).contiguous()
        x = x.to(net.device)
        X, y = x[:-1, :], x[1:, :]
        y=y.view(-1)
        #print("x",X.size())
        #print(net)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        loss_sum += l
        updater.zero_grad()
        l.backward()
        updater.step()
        cnt+=1
    return loss_sum/cnt

def train(net, train_iter, vocab, lr, num_epochs):
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(num_epochs):
        loss_=train_epoch(net, train_iter, loss, updater)
        print('epoch: %d,loss: %f' % (epoch, loss_))
        #test(net, train_iter, loss)
        #if (epoch + 1) % 3 == 0:
            #print(sample.sample(batch_size, "春".decode('UTF-8')))

train(model,train_iter, vocab, lr, num_epochs)
torch.save(model, 'poetry-gen.pt')
