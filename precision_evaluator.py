import pandas as pd

df = pd.read_csv("data/sheet1.csv", delimiter=';')
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

min_output = min(y)
max_output = max(y)


def precision_factor(MSE_loss):
    return MSE_loss/(max_output - min_output)
