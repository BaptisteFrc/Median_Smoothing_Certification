import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import os

# set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        # Passage dans le RNN
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # Passage dans la couche linéaire
        out = self.fc(out[:, -1, :])

        return out


def NN_to_function_v2(model):
    def inner(input):
        model.eval()
        input = torch.FloatTensor(input).reshape(-1, 1).unsqueeze(0)
        pred = model(input)
        predicted_value = pred[0, -1].item()
        return predicted_value
    return inner


def load_model_v2():
    model = Model()
    model.load_state_dict(torch.load("models_neural_network/model_v2.pth"))
    return model


test = NN_to_function_v2(load_model_v2())
print(test([0 for i in range(24)]))

if __name__ == "__main__":

    # import data
    data = pd.read_csv('data/household_power_consumption.txt', sep=';', parse_dates={'datetime': [
        'Date', 'Time']}, infer_datetime_format=True, index_col='datetime', na_values=['?'])
    data = data.dropna().reset_index()
    data = data.set_index(pd.DatetimeIndex(
        data['datetime'])).drop('datetime', axis=1)
    data = data.resample('1h').fillna(method='bfill')[
        'Global_active_power'].reset_index(name='Global_active_power')
    # data.min : 0.078
    # data.max : 8.758

    def create_sequences(data: np.ndarray, sequence_length: int, pred_len: int = 1):
        """Create sequences from data with a given sequence (input) length and a given prediction (output) length

        Args:
            data (np.ndarray): data to create sequences from
            sequence_length (int): length of the input sequence
            pred_len (int, optional): length of the output sequence. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: input sequences
        """
        x = []
        y = []
        for i in range(len(data)-sequence_length-pred_len+1):
            x.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length:i+sequence_length+pred_len])
        return np.array(x), np.array(y)

    def preprocess_data(data, sequence_length, output_seq_len, split_ratio=0.8, batch_size=32):

        data = data['Global_active_power'].values.reshape(-1, 1)

        # Normalize the 'Global_active_power' data with MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # data = scaler.fit_transform(data)

        # pass output_seq_len here
        x, y = create_sequences(data, sequence_length, output_seq_len)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=1-split_ratio, shuffle=False)

        # Convert to Tensors and create data loaders
        train_data = TensorDataset(
            torch.Tensor(x_train), torch.Tensor(y_train))
        test_data = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=True, batch_size=batch_size)

        return train_loader, test_loader

    def train_model(model, train_loader, test_loader, num_epochs, learning_rate, patience=10):
        """Train the model and print the loss for each epoch

        Args:
            model: the model to train
            train_loader: the training data loader
            test_loader: the testing data loader
            num_epochs: the number of epochs to train
            learning_rate: the learning rate
            patience: the number of epochs to wait before early stopping
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(
        ), max_norm=1)  # clip gradients to prevent exploding gradient problem
        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        # adam optimizer (backward pass)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loss_list = []
        test_loss_list = []

        best_loss = float('inf')
        no_improve_epoch = 0

        for epoch in tqdm.tqdm(range(num_epochs), desc='Training the model', unit='epoch', total=num_epochs):

            # Training
            train_loss = 0
            for inputs, targets in train_loader:  # for each training step
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)  # forward pass
                targets = targets[:, -1, :]
                loss = criterion(output, targets)

                optimizer.zero_grad()  # clear the gradients
                loss.backward()  # backward pass
                optimizer.step()  # optimize the weights

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_loss_list.append(train_loss)

            # Testing
            test_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    output = model(inputs)
                    targets = targets[:, -1, :]
                    loss = criterion(output, targets)

                    test_loss += loss.item()

            test_loss /= len(test_loader)
            test_loss_list.append(test_loss)

            print('Epoch [{}/{}], Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch +
                                                                          1, num_epochs, train_loss, test_loss))

            # check for early stopping (prevent overfitting)
            if test_loss < best_loss:
                best_loss = test_loss
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1
                if no_improve_epoch == patience:
                    print('Early stopping')
                    break

        # Plot training and testing loss
        plt.plot(train_loss_list, label='Training Loss')
        plt.plot(test_loss_list, label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss Over Time')
        plt.legend()
        plt.show()

        # Save model to file
        torch.save(model.state_dict(), 'model_v2.pth')

        return model

    # Set the input sequence length to one week (168 hours)
    input_seq_len = 24  # time steps
    output = 1  # predict one time step into the future

    # Initialize the modek
    model = Model(1, 128,  1, output)
    # Move the model to the specified device (CPU or GPU)
    model = model.to(device)

    # Set the number of training epochs and learning rate
    num_epochs = 20
    learning_rate = 0.001

    # Process data
    train_loader, test_loader = preprocess_data(
        data, input_seq_len, output, batch_size=64)

    # Train the model
    model = train_model(model, train_loader, test_loader,
                        num_epochs, learning_rate)
