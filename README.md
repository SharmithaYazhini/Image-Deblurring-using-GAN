## Image Deblurring with Generative Adversarial Networks (GANs)
This repository contains an implementation of a Generative Adversarial Network (GAN) for image deblurring. The goal is to train a deep learning model that can remove blur from images, producing sharp and clear output images.

## Dataset
The model is trained on the GoPro dataset, which consists of pairs of blurred and sharp images captured with a GoPro camera. The dataset is divided into training and testing sets.

## Model Architecture
The GAN architecture consists of two main components:

## Generator: 
The generator takes a blurred image as input and generates a deblurred (sharp) version of the image. The generator architecture is based on an encoder-decoder structure with residual blocks and skip connections.

## Discriminator: 
The discriminator is a convolutional neural network that takes pairs of images (blurred and sharp/generated) as input and tries to classify them as real (sharp) or fake (generated). In this implementation, a Patch-GAN discriminator is used, which classifies individual patches of the image instead of the entire image.

## Loss Functions
The generator is trained with a combination of two loss functions:

## Adversarial Loss: 
This loss encourages the generator to produce images that can fool the discriminator into classifying them as real (sharp).
## Perceptual Loss: 
This loss compares the high-level features of the generated image and the ground truth sharp image, using a pre-trained VGG-16 network. This helps the generator produce sharper and more realistic images.

The discriminator is trained with the Wasserstein GAN loss function.

## Results

![image](https://github.com/SharmithaYazhini/Image-Deblurring-using-GAN/assets/104150250/0ba6abca-a7cf-45c2-8b05-cb2eccd4eb7e)


## Usage
To train the model, run the fit() function with the appropriate dataset and number of steps. The model checkpoints will be saved every 5,000 steps in the training_checkpoints directory.
To test the model, use the generate_images() function and provide a blurred input image. The function will generate and display the deblurred output alongside the input and ground truth sharp images.

## Requirements

TensorFlow
Python 3.x
Other Python libraries (e.g., NumPy, Matplotlib)

Please take a look at the code for specific library versions and installation instructions.

Link to the trained model: https://drive.google.com/drive/folders/1IlTWdhhaBnxLJUsK0-nSCPEc3W1fowgp
