# slide: explication d'une attaque et de la FGSM
# inspiré de https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
#  We construct the image adversary by calculating the gradients of the loss, computing the sign of the gradient, and then using the sign to build the image adversary
from regression_model import load_model
import torch
from torch import nn
from math import sqrt


model = load_model()
input = [[[17.76, 42.42, 1009.09, 66.26], [468.27]]]


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    print(sign_data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + sqrt(epsilon)/2*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = nn.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def attack_1(model, input, epsilon):

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
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        attacked_output = model(perturbed_data)

    # Return the accuracy and an adversarial example
    return perturbed_data.tolist(), output.tolist(), attacked_output.tolist()


print(attack_1(model, input, 0.5))
