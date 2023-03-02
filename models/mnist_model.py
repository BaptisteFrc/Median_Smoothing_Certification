import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1)


train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                         batch_size=128,
                                         shuffle=True,
                                         num_workers=1),

    'test': torch.utils.data.DataLoader(test_data,
                                        batch_size=128,
                                        shuffle=True,
                                        num_workers=1),
}


def train(num_epochs, model, loaders):

    model.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = model(b_x)
            loss = F.nll_loss(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        pass
    pass
    torch.save(model.state_dict(), "mnist.pt")


def test():
    # Test the model
    model.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
        print('Test Accuracy of the model on the 10000 test images: %.10f' % accuracy)

    pass


def pre_image(img_path, model):
    img = Image.open(img_path)
    img = transforms.Grayscale()(img)
    transform_norm = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((28, 28))])
    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    # input = Variable(image_tensor)
    plt.imshow(img_normalized[0].permute(1, 2, 0), cmap="gray")
    with torch.no_grad():
        model.eval()
        output = model(img_normalized)
        pred_y = torch.max(output, 1)
        return pred_y


if __name__ == '__main__':
    model = NeuralNetwork()
    num_epochs = 3
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #train(num_epochs, model, loaders)
    # test()
    model.load_state_dict(torch.load("saved_model/mnist.pt"))
    print(pre_image("data/7n.png", model))
