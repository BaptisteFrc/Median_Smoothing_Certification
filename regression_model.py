import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tqdm
import copy
import numpy as np

# changer la taille du model, tracer training loss, relativement, verifier que ca n'overfit pas

"""
looking for when the nn will not give satisfying results...
modèle 1: 4,128,64,8,1 
modèle 2: 4,128,8,1
modèle 3: 4,64,8,1
modèle 4: 4,16,8,1
modèle 5: 4,4,8,1
modèle 6: 4,4,4,1
modèle 7: 4,4,1
modèle 8: 4,1
... it always does appparently

Tries to have a better loss...
... unconclusive.
"""


# Import data
'''
The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011),
when the power plant was set to work with full load.
Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V)
to predict the net hourly electrical energy output (EP)  of the plant.
'''
df = pd.read_csv("data/sheet1.csv", delimiter=';')
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
X_train = torch.tensor(X_train.values, dtype=torch.float64)
y_train = torch.tensor(y_train.values, dtype=torch.float64).reshape(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float64)
y_test = torch.tensor(y_test.values, dtype=torch.float64).reshape(-1, 1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 1)

        #self.dropout1 = nn.Dropout(0.0001)
        #self.dropout2 = nn.Dropout(0.0001)
        #self.dropout3 = nn.Dropout(0.0001)
        self.double()

    def forward(self, x):
        x = self.fc1(x)
        #x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        #x = self.dropout3(x)
        y_pred = self.fc4(x)
        return y_pred


# parametres
batch_size = 64
num_epochs = 250
learning_rate = 0.0005

model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train():
    best_mse = np.inf
    train_loss = []
    test_loss = []
    batch_start = torch.arange(0, len(X_train), batch_size)
    model.train()
    for epoch in range(num_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
            train_loss.append(float(loss))

        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        test_loss.append(mse.detach().numpy())
    #     if mse < best_mse:
    #         best_mse = mse
    #         best_weights = copy.deepcopy(model.state_dict())
    # model.load_state_dict(best_weights)
    torch.save(model.state_dict(), "regression.pt")
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % np.sqrt(mse.detach().numpy()))
    plt.scatter(np.arange(num_epochs), train_loss, label='train_loss', s=10)
    plt.scatter(np.arange(num_epochs), test_loss, label='test_loss', s=10)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE(log)')
    plt.title("MSE: %.2f" % mse+" RMSE: %.2f" % np.sqrt(mse.detach().numpy()))
    plt.legend()
    plt.show()


def test(X):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("regression.pt"))
    model.eval()
    X = torch.DoubleTensor(X)
    y_pred = model(X)
    return y_pred.item()


train()
# tes1 = [14.96, 41.76, 1024.07, 73.17]
# tes2 = [463.26]
# [17.76, 42.42, 1009.09, 66.26]
# [468.27]
print(test([17.76, 42.42, 1009.09, 66.26]))
