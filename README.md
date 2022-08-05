# Pizza Ice Cream CNN

## The Neural Network

This convolutional neural network predicts whether an image is of pizza or ice cream. The model will predict a value close to 0 if the image is predicted to be of pizza and a 1 if the image is predicted to be of ice cream. The model uses the pretrained VGG16 base provided by Keras (these layers are untrained in the model) and — since the model only predicts binary categorical values — uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard SGD optimizer with a learning rate of 0.001 and a dropout layer to prevent overfitting. The model has an architecture consisting of:
- 1 Horizontal random flip layer (for image preprocessing)
- 1 VGG16 base model (with an input shape of (128, 128, 3))
- 1 Flatten layer
- 1 Dropout layer (with a dropout rate of 0.3)
- 1 Hidden layer (with 256 neurons and a ReLU activation function
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Note that when running the **pizza_icecream_cnn.py** file, you will need to input the paths of the training, testing, and validation sets as strings — the location for where to put the paths are signified in the file with the words "< PATH TO IMAGES THAT DEPICT A TUMOR >" and "< PATH TO IMAGES THAT DO NOT DEPICT A TUMOR >."

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection. Credit for the dataset collection goes to **def me(x)**, **Loai abdalslam**, **Anshu Sinha**, and others on *Kaggle*. Note that the images from the original dataset are resized to 128 x 128 images so that they are more maneagable for the model. They are considered RGB by the model (the images have three color channels) because the VGG16 model only accepts images with three color channels. The dataset is not included in the repository because it is too large to stabley upload to Github, so just use the link about to find and download the dataset.

## Libraries
This neural network was created with the help of the Tensorflow and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
