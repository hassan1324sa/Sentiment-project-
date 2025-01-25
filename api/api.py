from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
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



app = FastAPI(title="Text Sentiment API")

DB = [
    {"id": 1, "text": "the product is very good ", "sentiment": None},
]


class Text(BaseModel):
    text: str = Field(title="The text to analyze", min_length=2, max_length=100)
    sentiment: bool

@app.get("/")
async def get():
    if not DB:
        raise HTTPException(status_code=404, detail="No texts found")
    sentence = DB[0]["text"]
    feel = predict(sentence) 
    print(feel)
    updated_text = {"id": DB[0]["id"], "text": sentence, "sentiment": feel}
    DB[0] = updated_text
    return { "data": DB}

@app.post("/addText")
async def add_text(text: Text):
    new_entry = {"id": len(DB) + 1, "text": text.text, "sentiment": text.sentiment}
    DB.append(new_entry)
    return {"data": new_entry}


@app.put("/putText/{text_id}")
async def update_text(text_id: int, text: Text):
    found_entry = next((entry for entry in DB if entry["id"] == text_id), None)
    if found_entry:
        found_entry["text"] = text.text
        found_entry["sentiment"] = text.sentiment
        return { "data": found_entry}
    else:
        raise HTTPException(status_code=404, detail="Text not found")


@app.delete("/deleteText/{text_id}")
async def delete_text(text_id: int):
    found_entry = next((entry for entry in DB if entry["id"] == text_id), None)
    if found_entry:
        DB.remove(found_entry)
        return {"message": "Text deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Text not found")