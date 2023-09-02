from rednet import *
from datasets import Dataset


class RedNetTrainer():
    def __init__(self, train_data, test_data, kernel_size, strides, padding, model_name,                   
                 num_blocks=5, layers_per_block=2, filters=64,  activation="relu", 
                 learning_rate=0.001, batch_size=32, epochs=10, patch_size=50, 
                 padding_type="constant", gaussian_noise_mean=0, gaussian_noise_stddev=1,
                 use_padding=True) -> None:
        
        self.train_data = train_data
        self.test_data = test_data
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.model_name = model_name
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        self.filters = filters
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patch_size = patch_size
        self.padding_type = padding_type
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_stddev = gaussian_noise_stddev
        self.use_padding = use_padding

        self.model = None
        self.history = None
        self.patch_train_set = None
        self.patch_test_set = None
    

    def train_model(self, load_model=False):
        self.patch_train_set, self.patch_test_set = self.get_datasets()
        in_shape = (self.patch_train_set[0].shape[1], self.patch_train_set[0].shape[2], self.patch_train_set[0].shape[3])

        model_checkpoint = self.get_model_checkpoint()

        if (load_model):
            self.model = tf.keras.models.load_model(self.model_name)
        else:
            rednet_instance = RedNet(input_shape=in_shape, num_blocks=self.num_blocks, 
                                    layers_per_block=self.layers_per_block, filters=self.filters, 
                                    kernel_size=self.kernel_size, strides=self.strides, 
                                    padding=self.padding, activation=self.activation)       
        
            self.model = rednet_instance.get_rednet()

        self.model.summary()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), 
                           loss=tf.keras.losses.MeanSquaredError())
        
        self.history = self.model.fit(x=self.patch_train_set[0], y=self.patch_train_set[1], 
                                      validation_data=self.patch_test_set, batch_size=self.batch_size, 
                                      epochs = self.epochs, callbacks=[model_checkpoint], verbose=1,
                                      shuffle=True)
    

    def get_datasets(self):
        dataset = Dataset(train_data=self.train_data, test_data=self.test_data, patch_size=self.patch_size,
                          padding_type=self.padding_type, gaussian_noise_mean=self.gaussian_noise_mean,
                          gaussian_noise_stddev=self.gaussian_noise_stddev, use_padding=self.use_padding)
        
        return dataset.get_patch_train_test()
    

    def get_model_checkpoint(self):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_name, save_weights_only=False,
            monitor="val_loss", mode="min", save_best_only=True
        )

        return model_checkpoint_callback
