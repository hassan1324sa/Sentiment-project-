from torch import tensor ,device ,cuda , optim 
from torch.utils.data import Dataset, DataLoader
from torch import nn

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt




learningRate = 0.0001
epochs = 400
batchSize = 32

data = pd.read_csv("/home/hassan/AI/tensorflow/learning RNN/example2/Sentiment.csv")
data = data[['text','sentiment']]

data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


for idx , i in data.iterrows():
    i[0] = i[0].replace('rt',' ')

maxLen = 2000
tokenizer = Tokenizer(num_words=maxLen, split=' ')
tokenizer.fit_on_texts(data['text'].values)
x = tokenizer.texts_to_sequences(data['text'].values)
x = pad_sequences(x)



encoder = LabelEncoder()
y = encoder.fit_transform(data["sentiment"].values.reshape(-1,1))
# y


class Data(Dataset):
    def __init__(self, data, labels):
        self.data = tensor(data)
        self.labels = tensor(labels)
        self.len = len(data)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    


dataSets = Data(x,y)

trainLoader = DataLoader(dataSets,batch_size=32,shuffle=True,drop_last=True,)
# next(iter(trainLoader))[1]


class Model(nn.Module):
    def __init__(self, embedInputs ,outputSize,numLayers=1,hiddenSize=128):
        super().__init__()
        self.embed = nn.Embedding(embedInputs,hiddenSize)
        self.lstm = nn.LSTM(hiddenSize,hiddenSize,numLayers,batch_first=True,dropout=0.5)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hiddenSize,outputSize)

    def forward(self,x):
        x = self.dropout(self.embed(x))
        out,(hidden,mem) = self.lstm(x)
        return self.fc(hidden[-1,:,:]) 


device = device("cuda" if cuda.is_available() else 'cpu')

hidden_size = 64
num_layers = 3


model = Model(embedInputs=maxLen, outputSize=2,numLayers=num_layers, hiddenSize=hidden_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=learningRate)

loss_fn = nn.CrossEntropyLoss()

training_loss_logger = []
test_loss_logger = []
training_acc_logger = []
test_acc_logger = []




train_acc = 0

for epoch in range(0,epochs):

    
    # Set model to training mode
    model.train()
    steps = 0
    
    # Iterate through training data loader
    for x, y in trainLoader:
        bs = y.shape[0]
    
        y = y.to(device)
        x = x.to(device)
        
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (pred.argmax(1) == y).sum()
        steps += bs
    training_loss_logger.append(loss.item())
    train_acc = (train_acc/steps).item()
    training_acc_logger.append(train_acc)


print(training_acc_logger[-1])


plt.plot(range(epochs),training_acc_logger,"b",label= "Training Accuracy")

plt.ylabel("acc")
plt.xlabel("epochs")
plt.legend()


cuda.empty_cache()




# print(f"Predicted Sentiment: {prediction}")

# torch.save(model.state_dict(),"model.pth")