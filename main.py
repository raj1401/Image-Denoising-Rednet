from rednet_trainer import *
import numpy as np
import matplotlib.pyplot as plt
from denoiser import Denoiser


class Main:
    def __init__(self) -> None:
        pass

    def train_denoiser(self, normalized_train_data, normalized_test_data, model_name, load_model):
        """
        normalized_train_data and normalized_test_data should be 4-dimensional
        numpy arrays:
        first dim - number of images
        second and third dim - resolution of each image
        fourth dim - number of channels
        Each entry of these arrays should be between 0 and 1
        load_model is True if a previously saved model's training needs to be resumed
        otherwise False
        """
        kernel_size = 3
        strides = 1
        padding = "valid"

        # Other parameters can be accessed and changed
        rednet_trainer = RedNetTrainer(train_data=normalized_train_data, test_data=normalized_test_data,
                                       kernel_size=kernel_size, strides=strides, padding=padding,
                                       model_name=model_name, 
                                       filters=64, epochs=2, gaussian_noise_mean=0.2, gaussian_noise_stddev=0.2,
                                       num_blocks=10, use_padding=True)
        rednet_trainer.train_model(load_model=load_model)

        return rednet_trainer
    

    def use_denoiser(self, denoising_model, images_list):
        denoiser = Denoiser(denoising_model=denoising_model, images_list=images_list)
        return denoiser.get_denoised_images()



if __name__ == "__main__":
    main = Main()
    bsds500_path = "bsds500_numpy_data"
    model_name = "bsds500_rednet10_64_filters_padded"

    load_data_file = np.load(os.path.join(bsds500_path,"data.npz"))

    ##################################### Training on BSDS500 Dataset #####################################

    # training_data = load_data_file["training_data"]
    # validation_data = load_data_file["validation_data"]

    # training_data = (training_data / 255.0).astype("float32")
    # validation_data = (validation_data / 255.0).astype("float32")

    # load_model = False    

    # rednet_trainer = main.train_denoiser(normalized_train_data=training_data, normalized_test_data=validation_data,
    #                                      model_name=model_name, load_model=load_model)


    #################################### Testing on BSDS500 Dataset ###########################################

    testing_data = load_data_file["testing_data"]

    testing_data = (testing_data / 255.0).astype("float32")

    noisy_testing_data = np.clip(testing_data + np.random.normal(0.2,0.2,testing_data.shape),0,1)
    model = tf.keras.models.load_model(model_name)
    imgs_list = []
    for i in range(2):
        imgs_list.append(noisy_testing_data[i,:])
    
    denoised_imgs_list = main.use_denoiser(denoising_model=model, images_list=imgs_list)

    for i in range(2):
        plt.subplot(1,3,1)
        plt.imshow(testing_data[i,:])
        plt.title("Original Image")

        plt.subplot(1,3,2)
        plt.imshow(imgs_list[i])
        plt.title("Noisy Image")

        plt.subplot(1,3,3)
        plt.imshow(denoised_imgs_list[i] / np.max(denoised_imgs_list[i]))
        plt.title("Denoised Image")

        plt.show()
