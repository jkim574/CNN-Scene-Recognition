# CNN-for-scene-recognition
## 1. Setup a Python Virtual Environment: 

python3 -m venv --system-site-packages ~/PyTorch
Here the name of our virtual environment is PyTorch (you can use any other name if you want).

2. Activate the environment:

source ~/PyTorch/bin/activate
This will activate your virtual environment and your shell prompt is prefixed with (PyTorch).

I built my own network with basic building block for neural networks.

A Flatten layer to convert the 2D pixel array to a 1D array of numbers
A Dense layer with 128 nodes and a ReLU activation.
A Dense layer with 64 nodes and a ReLU activation.
A Dense layer with 10 nodes.

Dataset

Dataset is MiniPlaces that has 120K images from 100 scene categories. The categories are mutually exclusive. The dataset is split into 100K images for training, 10K images for validation, and 10K for testing.

The original image resolution for images in MiniPlaces is 128x128. To make the training feasible, data loader reduces the image resolution to 32x32. Our data loader will also download the full dataset the first time you run train_miniplaces.py. 

An implementation of LeNet-5 is provided as SimpleConvNet class in student_code.py. You will modify this class to implement your own deep network. You will also re-use your implementation of train_model and test_model in Part 2. Please avoid changing any other files besides student_code.py except for profiling purpose in train_miniplaces.py (see Profiling Your Model).

When you run train_miniplaces.py, the python script will save two files in the "outputs" folder

checkpoint.pth.tar is the model checkpoint at the latest epoch.
model_best.pth.tar is the model weights that has highest accuracy on the validation set. You will need to submit this file together with student_code.py.


