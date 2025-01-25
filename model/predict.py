from torch import nn ,no_grad ,softmax ,load ,tensor,long
from re import sub 
from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 

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


def predict(sentence,device="cuda"):
    model = Model(embedInputs=2000, outputSize=2,numLayers=3, hiddenSize=64).to(device)
    model.load_state_dict(load("model/model.pth"))
    model.eval()
    tokenizer = Tokenizer(num_words=2000, split=' ')
    sentence = sentence.lower()
    sentence = sub('[^a-zA-z0-9\s]', '', sentence)
    sentence = sentence.replace('rt', ' ')
    sequence = tokenizer.texts_to_sequences([sentence])
    
    padded_sequence = pad_sequences(sequence, maxlen=2000)
    
    tensors = tensor(padded_sequence, dtype=long).to(device)
    with no_grad():
        output = model(tensors)
        prob = softmax(output, dim=1)
        predicted_class = prob.argmax(1).item()
    
    sentiment = True if predicted_class == 1 else False
    return sentiment


# BASE_URL = "https://sentiment-analysis-api-production-afe1.up.railway.app/"
# response = get(f"{BASE_URL}/")

# sentence= response.json()["data"][0]["text"]
# Feel = predict(sentence)
# updated_text = {"text":sentence,"sentiment": Feel}
# response = put(f"{BASE_URL}/putText/1", json=updated_text)