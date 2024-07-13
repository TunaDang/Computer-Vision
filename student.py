# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
# import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# END IMPORTS

#########################################################
###              BASELINE MODEL
#########################################################

class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 6, 3, 2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 12, 3, 2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(12, 24, 3, 2, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(24 * 8 * 8, 128)
        self.relu4 = nn.ReLU()
        self.cls = nn.Linear(128, num_classes)
        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 24 * 8 * 8).float()
        x = self.fc(x)
        x = self.relu4(x)
        x = self.cls(x)
        # TODO-BLOCK-END
        return x

def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    model = net
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss = loss.item()
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    _, predicted = torch.max(outputs, 1)
    num_correct = (predicted == labels).sum()
    total_images = labels.size(0)
    # TODO-BLOCK-END

    return running_loss, num_correct, total_images
#########################################################
###             STUDENT MODEL
#########################################################

class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        # Key insight: Use batch normalization, pooling, and dropout to prevent overfitting vs baseline
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)

        self.cls = nn.Linear(512, num_classes)
        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        x = x.view(-1, 256 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)
        x = self.cls(x)
        # TODO-BLOCK-END
        return x

#########################################################
###             ADVERSARIAL IMAGES
#########################################################

def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_image   (torch Tensor, same dimensions as img)
                        adversarial image
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    return perturbed_image, noise

