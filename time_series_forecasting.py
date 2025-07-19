import yfinance as yf
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = yf.download("UAL",start="2015-01-01",end="2025-01-01")

data = df[["Close"]].values

mm = MinMaxScaler(feature_range=(0,1))
scaled_data = mm.fit_transform(data)

def create_sequences(data, seq_length=60):
    X, Y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  # Past 60 days
        Y.append(data[i])               # Next day
    return np.array(X), np.array(Y)

sequence_length = 60
x, y = create_sequences(scaled_data, sequence_length)

x = x.astype(np.float32)
y = y.astype(np.float32)
print("X shape:", x.shape)  # (samples, 60, 1)
print("y shape:", y.shape)  # (samples, 1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

class NeuralNetwork(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,num_layers=2,output_size=1):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.2)
        self.l1 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        hidden_states =torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        cell_states =torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        out,_ = self.lstm(x,((hidden_states,cell_states)))
        out = self.l1(out[:,-1,:])
        return out

class CustomDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

def train_the_model(model, x_train, x_test, y_train, y_test):
    batch_size = 32
    train = DataLoader(CustomDataset(x_train, y_train), shuffle=True, batch_size=batch_size)
    test = DataLoader(CustomDataset(x_test, y_test), shuffle=False, batch_size=batch_size)

    num_epochs = 100
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    min_loss = np.inf
    best_weights = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for (x, y) in train:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / len(train):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for (x, y) in test:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()

        if val_loss < min_loss:
            min_loss = val_loss
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)
    print(f"Best Validation Loss: {min_loss:.4f}")
            
model = NeuralNetwork().to(device)
train_the_model(model,x_train,x_test,y_train,y_test)
torch.save(model.state_dict(), "stock_price_lstm.pth")
