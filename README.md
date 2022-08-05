# Pizza Ice Cream CNN

## The Neural Network

This convolutional neural network predicts whether an image is of pizza or ice cream. The model will predict a value close to 0 if the image is predicted to be of pizza and a 1 if the image is predicted to be of ice cream. The model uses the pretrained VGG16 base provided by Keras (these layers are untrained in the model) and — since the model only predicts binary categorical values — uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard SGD optimizer with a learning rate of 0.001 and a dropout layer to prevent overfitting. The model has an architecture consisting of:
- 1 Horizontal random flip layer (for image preprocessing)
- 1 VGG16 base model (with an input shape of (128, 128, 3))
- 1 Flatten layer
- 1 Dropout layer (with a dropout rate of 0.3)
- 1 Hidden layer (with 256 neurons and a ReLU activation function
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Note that when running the **pizza_icecream_cnn.py** file, you will need to input the paths of the training, testing, and validation sets as strings — the location for where to put the paths are signified in the file with the words "< PATH TO TRAINING DATA >," "< PATH TO TESTING DATA >," and "< PATH TO VALIDATION DATA >." Note that when you input these paths, they should be such that — when they are concatenated with the individual elements listed in the **path_list** variable — they are complete paths. For example:
> The dataset is stored in a folder called *food-data*, under which are the respective *train*, *test*, and *valid* directories that can be downloaded from the source (the link to the download site is below)
> - Thus, your file stucture is something like:

>     ↓ folder1
>       ↓ folder2
>         ↓ food-data
>           ↓ train
>             ↓ pizza
>                 < Images >
>             ↓ icecream
>                 < Images >
>           ↓ test
>             ↓ pizza
>                 < Images >
>             ↓ icecream
>                 < Images >
>           ↓ valid
>             ↓ pizza
>                 < Images >
>             ↓ icecream
>                 < Images >

> The paths you input should be something along the lines of: *~/folder1/folder2/food-data/train/*, *~/folder1/folder2/food-data/test/*, and *~/folder1/folder2/food-data/valid/*, and your **path_list** should be set to ['pizza', 'icecream'], so that when the **create_dataset()** function is running it concatenates the paths with the elements of **path_list** to produce fully coherent paths, such as *~/folder1/folder2/food-data/train/pizza*, *~/folder1/folder2/food-data/train/icecream*, *~/folder1/folder2/food-data/test/pizza*, etc.

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
