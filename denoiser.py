import numpy as np


class Denoiser:
    def __init__(self, denoising_model, images_list) -> None:
        self.model = denoising_model
        self.images_list = images_list

        self.model_config = denoising_model.get_config()
        self.input_shape = self.model_config["layers"][0]["config"]["batch_input_shape"]
        self.patch_size = self.input_shape[1]


    def get_denoised_images(self):
        denoised_images_list = []
        print("Denoising Images")
        for img_idx in range(len(self.images_list)):
            print(f"Denoised Image(s) {img_idx} of {len(self.images_list)}", end="\r")
            denoised_images_list.append(self.denoise_image(self.images_list[img_idx]))
            #denoised_images_list.append(self.advanced_denoise(self.images_list[img_idx]))
        
        return denoised_images_list
    

    def denoise_image(self, image_array):
        img_length, img_breadth = image_array.shape[0], image_array.shape[1]

        # Padding the original image if necessary
        if (img_length % self.patch_size != 0):
            deficit_length = self.patch_size - (img_length % self.patch_size)
        else:
            deficit_length = 0
        
        if (img_breadth % self.patch_size != 0):
            deficit_breadth = self.patch_size - (img_breadth % self.patch_size) 
        else:
            deficit_breadth = 0
        
        pad_values = ((deficit_length//2, deficit_length - deficit_length//2), 
                      (deficit_breadth//2, deficit_breadth - deficit_breadth//2), (0,0))
        
        padded_image = np.pad(image_array, pad_width=pad_values, mode="constant")
        padded_denoised_image = np.zeros(shape=padded_image.shape)

        num_strides_length = padded_image.shape[0] // self.patch_size
        num_strides_breadth = padded_image.shape[1] // self.patch_size

        for i in range(num_strides_length):
            l_idx = i * self.patch_size
            for j in range(num_strides_breadth):                
                b_idx = j * self.patch_size
                denoised_output = self.model.predict(np.expand_dims(padded_image[l_idx:l_idx+self.patch_size, 
                                                                     b_idx:b_idx+self.patch_size, :],axis=0),
                                                                     verbose=0)
                
                padded_denoised_image[l_idx:l_idx+self.patch_size, 
                                      b_idx:b_idx+self.patch_size, :] = denoised_output[0,:]
        
        unpadded_denoised_image = padded_denoised_image[deficit_length//2:deficit_length//2 + img_length,
                                                        deficit_breadth//2:deficit_breadth//2 + img_breadth, :]
        
        return unpadded_denoised_image
    

    def advanced_denoise(self, image_array):
        initial_denoised_image = self.denoise_image(image_array=image_array)
        offset_amount = self.patch_size // 2

        offset_denoised_image = self.denoise_image(image_array=initial_denoised_image[offset_amount:, offset_amount:, :])
        # Averaging
        initial_denoised_image[offset_amount:, offset_amount:, :] = 0.5 * (offset_denoised_image + 
                                                                           initial_denoised_image[offset_amount:, offset_amount:, :])
        return initial_denoised_image
