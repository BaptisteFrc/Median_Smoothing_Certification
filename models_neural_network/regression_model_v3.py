import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import os
from joblib import dump, load
from cleverhans.torch.attacks import projected_gradient_descent, fast_gradient_method
from adversarial_attacks.attack_FGSM import attack_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Définition du modèle LSTM


class Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=1, output_size=1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialisation de l'état caché
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).requires_grad_().to(device)

        # Passage dans le LSTM

        out, _ = self.lstm(x, (h0, c0))

        # Passage dans la couche linéaire
        out = self.fc(out[:, -1, :])

        return out


def NN_to_function_v2(model):
    def inner(input):
        scaler = load('models_neural_network/scaler.bin')
        model.eval()
        input = torch.FloatTensor(input).reshape(-1, 1)
        input = scaler.fit_transform(input)
        input = torch.FloatTensor(input).unsqueeze(0)

        pred = model(input)

        predicted_value = pred[0, -1].item()
        return scaler.inverse_transform([[predicted_value]])[0][0]
    return inner


def load_model_v2():
    model = Model()
    model.load_state_dict(torch.load(
        "models_neural_network/model_v3.pth", map_location=torch.device('cpu')))
    return model


# 50 premieres valeurs
l1 = [4.216, 2.79, 4.07, 3.206, 3.314, 3.464, 2.41, 1.044, 1.008,
      1.334, 2.542, 2.402, 2.294, 2.076, 0.222, 2.28, 2.94, 2.292,
      3.65, 1.962, 1.754, 1.744, 2.11, 2.818, 3.388]

l2 = [3.328, 4.078, 2.388, 3.576, 2.69, 1.116, 0.278, 0.296, 0.21,
      0.21, 0.378, 0.396, 0.214, 0.684, 2.128, 3.678, 1.352, 1.254,
      1.734, 1.48, 0.408, 1.888, 1.68, 1.946, 2.786]


#test = NN_to_function_v2(load_model_v2())
# print(test(l2[:24]))

input = l1
model = Model()
adv = attack_2(model, input, 1)
print(adv)
