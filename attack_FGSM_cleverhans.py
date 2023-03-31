from regression_model import load_model
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from cleverhans.torch.attacks.gradient import GradientAttack

model = load_model()

df = pd.read_csv("data/sheet1.csv", delimiter=';')
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True)

X_test = torch.tensor(X_test.values, dtype=torch.float64)
y_test = torch.tensor(y_test.values, dtype=torch.float64).reshape(-1, 1)


fgm = GradientAttack(
    model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=0.3)


X_adv = x_adv = fgm.perturb(X_test, y_test)
