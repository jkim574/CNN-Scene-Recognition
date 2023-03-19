# CNN-for-scene-recognition
1. Setup a Python Virtual Environment: 

python3 -m venv --system-site-packages ~/PyTorch
Here the name of our virtual environment is PyTorch (you can use any other name if you want).

2. Activate the environment:

source ~/PyTorch/bin/activate
This will activate your virtual environment and your shell prompt is prefixed with (PyTorch).

3. From your virtual environment shell, run the following commands to successfully upgrade pip and install PyTorch and other dependencies: 

## pip install --upgrade pip
## pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
## pip install tqdm
