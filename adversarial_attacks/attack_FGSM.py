'''
inspired by https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
We construct the image adversary by calculating the gradients of the loss, computing the sign of the gradient, and then using the sign to build the image adversary
'''
import torch
from torch import nn
from joblib import load


def attack_1(model, input):
    # FGSM attack compatible with regression_model

    # Loop over all examples in test set
    for data, target in input:
        data = torch.DoubleTensor(data)
        target = torch.DoubleTensor(target)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)

        # Calculate the loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        attack = data_grad.sign()

        # Re-classify the perturbed image
        attacked_output = model(data+attack)

    # Return the accuracy and an adversarial example
    return attack.tolist(), output.tolist(), attacked_output.tolist()


def attack_2(model, input):
    # FGSM attack compatible with regression_model_household

    scaler = load('models_neural_network/scaler.bin')
    input = torch.FloatTensor(input).reshape(-1, 1)
    input = scaler.fit_transform(input)
    target = torch.FloatTensor([input[-1]]).reshape(-1, 1)
    input = torch.FloatTensor(input[:-1, :]).unsqueeze(0)
    model.eval()
    # Set requires_grad attribute of tensor. Important for Attack
    data = input.clone().detach().requires_grad_(True)

    # Forward pass the data through the model
    output = model(data)
    #output = torch.FloatTensor(scaler.inverse_transform(output.detach().numpy()))

    # Calculate the loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass

    loss.backward()

    # Collect datagrad
    #input.requires_grad = True
    data_grad = data.grad.data

    # Call FGSM Attack
    attack = attack = data_grad.sign()

    # Re-classify the perturbed image
    attacked_output = model(data+attack)
    attacked_output = attacked_output[0, -1].item()
    attacked_output = scaler.inverse_transform([[attacked_output]])[0][0]

    output = output[0, -1].item()
    output = scaler.inverse_transform([[output]])[0][0]

    attack = attack.tolist()

    # Return the accuracy and an adversarial example
    return attack, output, attacked_output