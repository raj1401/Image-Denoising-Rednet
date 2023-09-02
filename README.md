# Image-Denoising-Rednet
 Implementing Convolutional AutoEncoders with Symmetric Skip Connections for Image Denoising

Project Organization:

**custom_layers.py** implements the building blocks needed to make the model\
**rednet.py** uses classes from **custom_layers.py** to implement the model\
**rednet_trainer.py** uses this model and trains it\
**denoiser.py** implements how one can use a trained model to denoise images\
**datasets.py** creates normalized, patched data for training the model\
**main.py** implements the Main class whose methods can be used to train a denoiser or use it

This project follows the model architecture pioneered by Mao et al. https://doi.org/10.48550/arXiv.1606.08921
