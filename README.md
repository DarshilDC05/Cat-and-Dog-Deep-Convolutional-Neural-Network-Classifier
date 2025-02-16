# Cat and Dog Deep Convolutional Neural Network Classifier

This repository contains the files associated with a Deep CNN Classifier that classifies input images to 2 classes : Cats and Dogs. This project was made using the `pytorch` framework.

## Hardware 

The training, preprocessing and testing of the model were done locally on a windows laptop with the following specs

- CPU : 13th Generation Intel® Core™ i7-13700H Processor
- GPU : NVIDIA GeForce® RTX™ 4050 6GB GDDR6 (55W)

## Dataset

The dataset directory consists of two folders : 

- training_set
- test_set

Each of these folder have two sub folders, `cats` and `dogs`. The training_set contains around 4000 samples of cats and dogs each, while the test_set contains about 1000 of each. The imaages are of various sizes and from different perspectives. This dataset was taken from Kaggle.

[Link to the dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

## Preprocessing and Data Augmentation

The images are resized to a `256 x 256` aspect ratio first and the augmented by flipping and rotating. Then they are center cropped to an aspect ratio of `224 x 224`. The RGB values of these images are then normalized to improve stability and converge the model faster during training. All functions used are the standard ones from pytorch library.


## Architecture

The model has 4 convolutional layers followd by 2 fully connected layers. Each convolutional layer has `Relu` activation and a pooling layer to reduce dimension. Batch normalization after each convolutional layer is also performed to speed up training and better stability and results. Finally, the convolutional layers extract 512 feature maps from the image, which is then fed to the fully connected layers with `Linear` activation as the convolutional layers have already introduced non-linearity. The last layer has only 2 neurons, each corresponding to an ouput class. This model was chosen as it performed well and gave good accuracy without overfitting the dataset. The summary of each layer is goven below.


| Layer Name      | Type              | Parameters                        | Output Shape          | Purpose |
|---------------|-----------------|---------------------------------|-----------------------|----------|
| **conv1**       | Conv2d (3→32)    | 3×3 kernel, padding=1          | (32, 224, 224)       | Extracts low-level features (edges, textures) from input images. |
| **batch_norm1** | BatchNorm2d(32)  | 32 learnable γ & β params      | (32, 224, 224)       | Normalizes activations, speeds up training, stabilizes learning. |
| **pool**    | MaxPool2d(2,2)   | No learnable params            | (32, 112, 112)       | Reduces spatial size, retains most important features. |
| **conv2**       | Conv2d (32→64)   | 3×3 kernel, padding=1          | (64, 112, 112)       | Captures more complex patterns like shapes & textures. |
| **batch_norm2** | BatchNorm2d(64)  | 64 learnable γ & β params      | (64, 112, 112)       | Normalizes activations, improving generalization. |
| **pool**    | MaxPool2d(2,2)   | No learnable params            | (64, 56, 56)         | Further reduces spatial dimensions while preserving key features. |
| **conv3**       | Conv2d (64→128)  | 3×3 kernel, padding=1          | (128, 56, 56)        | Detects more abstract features like object parts. |
| **batch_norm3** | BatchNorm2d(128) | 128 learnable γ & β params     | (128, 56, 56)        | Normalization for stable training and better convergence. |
| **pool**    | MaxPool2d(2,2)   | No learnable params            | (128, 28, 28)        | Downsamples feature maps to focus on important regions. |
| **conv4**       | Conv2d (128→256) | 3×3 kernel, padding=1          | (256, 28, 28)        | Extracts high-level abstract features. |
| **batch_norm4** | BatchNorm2d(256) | 256 learnable γ & β params     | (256, 28, 28)        | Ensures stable training, helps gradient flow. |
| **pool**    | MaxPool2d(2,2)   | No learnable params            | (256, 14, 14)        | Further reduces spatial size to retain only essential info. |
| **fc1**        | Linear (50176→512) | 512 neurons                    | (512)                | Fully connected layer to learn complex relationships. |
| **dropout**    | Dropout (p=0.5)   | No learnable params            | (512)                | Reduces overfitting by randomly dropping connections. |
| **fc2**        | Linear (512→2)    | 2 neurons (cat vs. dog)        | (2)                  | Final classification layer. |

## Training Loop

The following directory configuration was used during the training

```text
├───assets
├───test_set
│   └───test_set
│       ├───cats
│       └───dogs
└───training_set
    └───training_set
        ├───cats
        └───dogs
```

Standard DataLoaders from the pytorch library are used to load the data using **4 cpu cores** (Subject to system specs and if data loading is the bottleneck for training). The loss function (criterion) chosen is **CrossEntropy()** and **Adam's Algorithm** optimizer was used for cost function minimization instead of gradient descent. The initial **learning rate** was set to `0.0005` with the **number of epochs** being set to `20`. These values were chosen after multiple cycles of trial and error as they seemed to be optimum for this model and did not overfit the data. The dataset is split into batches for training. The **batch size** chosen for training was `64` (Subject to GPU specs, can go higher if V-RAM is not being utilized enough to speed up training).

After training is finished, the [model is saved](./cat_dog_cnn.pth). Now its ready for testing.

## Testing and Validation

[test.py](./test.py) is the file that gets the accuracies. It does a forward pass of each sample from the [test set](/test_set/test_set/) and compares it to the ground truth to compute the accuracy. This [model](./cat_dog_cnn.pth) achieved **87.69%** accuracy over the test set, and can classify unambiguous images of cats and dogs almost everytime. Some samples tested are stores in [this folder](./assets/). Notably, this [popular image](./assets/cry.png) gets classified wrong, which is a limitation of this model.

## Usage

To use this model to classify cats and dogs, you can adjust the path to the given image in [predict.py](./predict.py) in *line 58* to your required image, and the image will be plotted using matplotlib with the predicted output as the title.

