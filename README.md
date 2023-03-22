# CNN-for-scene-recognition
## Installation
1. Setup a Python Virtual Environment: 

```
python3 -m venv --system-site-packages ~/PyTorch
```

Here the name of our virtual environment is PyTorch (you can use any other name if you want).

2. Activate the environment:
```
source ~/PyTorch/bin/activate
```
This will activate your virtual environment and your shell prompt is prefixed with (PyTorch).

3. From your virtual environment shell, run the following commands to upgrade pip, install PyTorch and other dependencies:
```
pip install --upgrade pip
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm
```

4. For deactivating the virtual environment:
```
deactivate
```

5. Once you have set up the virtual environment, run the following command after activating the environment to save the list of installed packages:
```
pip freeze > setup_output.txt
```

## Designed a convolutional network for the MiniPlaces dataset

I built my own network with basic building block for neural networks.

Dataset is MiniPlaces that has 120K images from 100 scene categories. The categories are mutually exclusive. The dataset is split into 100K images for training, 10K images for validation, and 10K for testing. The original image resolution for images in MiniPlaces is 128x128. To make the training feasible, data loader reduces the image resolution to 32x32. The data loader will also download the full dataset the first time you run train_miniplaces.py. 

An implementation of LeNet-5 is provided as SimpleConvNet class in student_code.py. I modified this class to implement my own deep network.
- A Flatten layer to convert the 2D pixel array to a 1D array of numbers
- A Dense layer with 128 nodes and a ReLU activation.
- A Dense layer with 64 nodes and a ReLU activation.
- A Dense layer with 10 nodes.


checkpoint.pth.tar is the model checkpoint at the latest epoch.
model_best.pth.tar is the model weights that has highest accuracy on the validation set. 


