import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable as V
from torch import optim
import torchvision as tv
import torchvision.transforms as tf


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1,num_layers=2):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(  
            input_size,
            hidden_size,  
             num_layers,  
        )
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x,_= self.rnn(input)
        s,b,h= x.shape
        x= x.view(s*b,h)
        x= self.reg(x)
        x= x.view(s,b,-1)
        return x



ZX_df = pd.read_csv('./ZX10208_agr.csv')

ZX_df['dates'] = pd.to_datetime(ZX_df.dates)
ZX_df['price_per_sku'] = ZX_df.groupby('SKU').price_per_sku.ffill()

df_59567 = ZX_df[ZX_df.SKU == 59567]
train_df = df_59567[df_59567.dates <= pd.to_datetime('2019-04-01')]
test_df = df_59567[df_59567.dates > pd.to_datetime('2019-04-01')]

dataset = train_df.price_per_sku.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: (x-min_value) / scalar, dataset))

look_back = 5
data_X, data_Y = create_dataset(dataset, look_back)


train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]
x1 = np.arange(0,train_size)
x2 = np.arange(train_size,train_size + test_size)


train_X = train_X.reshape(-1, 1, look_back)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, look_back)
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
train=torch.zeros(len(train_x),1,look_back+1)
train[:,:,:look_back]=train_x
train[:,:,look_back:(look_back+1)]=train_y
train = torch.utils.data.DataLoader(
    train,
    batch_size=1,
    shuffle=False,
    num_workers=2
    )


rnn = RNN(look_back,4)
print(rnn)
optimizer=torch.optim.Adam(rnn.parameters(),lr=0.01)
criterion=nn.MSELoss()

for epoh in range(40):
    for i, data in enumerate(train, 0):
        var_x = V(data[:,:,:look_back])
        var_y = V(data[:,:,look_back:(look_back+1)])
        out = rnn(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoh + 1) % 1 == 0: 
        print('Epoch: {}, Loss: {:.5f}'.format(epoh + 1, loss.item()))

rnn_e = rnn.eval()

train_x = V(train_x)
outputs_train = rnn_e(train_x)
pred_train = outputs_train.view(-1).data.numpy()
inputs=V(test_x)
outputs=rnn_e(inputs)
predicted = outputs.view(-1).data.numpy()

print(predicted)
