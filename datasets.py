import numpy as np
from numba import njit


class Dataset:
    """
    This class creates cropped training and test images of 
    patch_size * patch_size dimensions from the input data.
    train_data and test_data are numpy arrays 
    of shape [num_images, img_res_x, img_res_y, num_chanels].
    mMake sure these arrays have normalized values (between 0 and 1).
    padding_type can be any of the types numpy.pad allows. Default
    is constant which pads zeros.
    """
    def __init__(self, train_data, test_data, patch_size=50, padding_type="constant", 
                 gaussian_noise_mean=0, gaussian_noise_stddev=1, use_padding=True) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.patch_size = patch_size
        self.padding_type = padding_type
        self.mean = gaussian_noise_mean
        self.stddev = gaussian_noise_stddev
        self.use_padding = use_padding
    

    def get_patch_train_test(self):
        noisy_train_data, noisy_test_data = self.add_noise()

        if (self.use_padding):
            # Adds Padding
            padded_train, padded_test, padded_noisy_train, padded_noisy_test = self.add_padding([self.train_data, 
                                                                                                self.test_data, 
                                                                                                noisy_train_data, 
                                                                                                noisy_test_data])
            
            patchy_train = self.create_patch_dataset(padded_train, self.patch_size).astype("float32")
            patchy_test = self.create_patch_dataset(padded_test, self.patch_size).astype("float32")
            patchy_noisy_train = self.create_patch_dataset(padded_noisy_train, self.patch_size).astype("float32")
            patchy_noisy_test = self.create_patch_dataset(padded_noisy_test, self.patch_size).astype("float32")        
        else:
            # Crops Images according to patch size
            cropped_train, cropped_test, cropped_noisy_train, cropped_noisy_test = self.crop_dataset([self.train_data,
                                                                                                    self.test_data,
                                                                                                    noisy_train_data,
                                                                                                    noisy_test_data])

            patchy_train = self.create_patch_dataset(cropped_train, self.patch_size).astype("float32")
            patchy_test = self.create_patch_dataset(cropped_test, self.patch_size).astype("float32")
            patchy_noisy_train = self.create_patch_dataset(cropped_noisy_train, self.patch_size).astype("float32")
            patchy_noisy_test = self.create_patch_dataset(cropped_noisy_test, self.patch_size).astype("float32")


        return [patchy_noisy_train, patchy_train], [patchy_noisy_test, patchy_test]


    def add_noise(self):
        gaussian_noise_train = np.random.normal(self.mean, self.stddev, size=self.train_data.shape)
        gaussian_noise_test = np.random.normal(self.mean, self.stddev, size=self.test_data.shape)

        return np.clip(self.train_data + gaussian_noise_train, 0, 1), np.clip(self.test_data + gaussian_noise_test, 0, 1)
    

    def add_padding(self, array_list):
        output_list = []

        for arr in array_list:
            img_length, img_breadth = arr.shape[1], arr.shape[2]

            if (img_length % self.patch_size != 0):
                deficit_length = self.patch_size - (img_length % self.patch_size)
            else:
                deficit_length = 0
            
            if (img_breadth % self.patch_size != 0):
                deficit_breadth = self.patch_size - (img_breadth % self.patch_size) 
            else:
                deficit_breadth = 0

            pad_values = ((0,0), (deficit_length//2, deficit_length - deficit_length//2), 
                          (deficit_breadth//2, deficit_breadth - deficit_breadth//2), (0,0))

            padded_array = np.pad(arr, pad_width=pad_values, mode=self.padding_type)
            output_list.append(padded_array)
        
        return output_list
    

    def crop_dataset(self, array_list):
        output_list = []

        for arr in array_list:
            img_length, img_breadth = arr.shape[1], arr.shape[2]
            length_stride, breadth_stride = img_length//self.patch_size, img_breadth//self.patch_size
            output_list.append(arr[:, :length_stride*self.patch_size, :breadth_stride*self.patch_size, :])
        
        return output_list
    
    
    @staticmethod
    @njit
    def create_patch_dataset(original_data, patch_size):
        original_length = original_data.shape[1]
        original_breadth = original_data.shape[2]

        num_strides_length = original_length // patch_size
        num_strides_breadth = original_breadth // patch_size

        total_patches = original_data.shape[0] * num_strides_length * num_strides_breadth

        patch_dataset = np.zeros(shape=(total_patches, patch_size, patch_size, original_data.shape[3]))
        patch_idx = 0

        for img_idx in range(original_data.shape[0]):
            for i in range(num_strides_length):
                l_idx = i * patch_size
                for j in range(num_strides_breadth):                    
                    b_idx = j * patch_size
                    patch_dataset[patch_idx,:,:,:] = original_data[img_idx, l_idx:l_idx+patch_size,
                                                                   b_idx:b_idx+patch_size,:]
                    patch_idx += 1
        
        return patch_dataset
