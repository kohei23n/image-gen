# 1. Importing Dependencies and Data
## bringing in tensorflow
import tensorflow as tf
import numpy as np
import os

## set memory growth for GPUs to avoid memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## bringing in matplotlib for visualisation
from matplotlib import pyplot as plt

## bringing in additional imports for custom dataset
from PIL import Image

## load custom images
def load_custom_dataset(data_path, img_size=(64, 64)):
    image_paths = []
    for file in os.listdir(data_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(data_path, file))
    
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)  # RGB instead of grayscale
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image)
    return dataset

## load custom dataset
ds = load_custom_dataset('data/frames', img_size=(64, 64))

# 2. Visualise Data and Build Dataset
## do some data transformation


## setup connection aka iterator
dataiterator = ds.as_numpy_iterator()

## scale images to [-1, 1] range for better GAN training
def scale_images(image):
    return (image * 2.0) - 1.0

## reload the dataset and apply preprocessing
ds = load_custom_dataset('data/frames', img_size=(64, 64))
## running the dataset through the scale_images preprocessing function
ds = ds.map(scale_images)
## augment the dataset with some random transformations
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # you can add more augmentations as needed
    return image
ds = ds.map(augment)
## cache the dataset for that batch
ds = ds.cache()
## shuffle it up (adjust number based on your dataset size)
ds = ds.shuffle(1000)  # Reduced from 60000 since you likely have fewer images
## batch into smaller batches due to larger images
ds = ds.batch(4)  # Reduced from 512 due to memory constraints
## reduces the likelihood of bottlenecking
ds = ds.prefetch(16)  # Reduced accordingly


# 3. Build Neural Network

## bring in the sequential api for the generator and discriminatoir
from keras.models import Sequential
## bringing in the layers for the neural network
from keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, BatchNormalization

def build_generator():
    model = Sequential()
    
    ## takes in random values and reshapes it to 8x8x256 for 64x64 output
    ## beginnings of a generated image
    model.add(Dense(8*8*256, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 256)))

    ## upsampling block 1 - 8x8 to 16x16
    model.add(UpSampling2D())
    model.add(Conv2D(128, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    ## upsampling block 2 - 16x16 to 32x32
    model.add(UpSampling2D())
    model.add(Conv2D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    ## upsampling block 3 - 32x32 to 64x64
    model.add(UpSampling2D())
    model.add(Conv2D(32, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    ## convolutional block for refinement
    model.add(Conv2D(32, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    ## conv layer to get to three channels (RGB) with tanh activation
    model.add(Conv2D(3, 3, padding='same', activation='tanh'))

    return model

generator = build_generator()

## generate new nostalgic images
img = generator.predict(np.random.randn(4, 128))
print(img.shape)

def build_discriminator():
    model = Sequential()

    ## convolutional block 1 - now expects 64x64x3 RGB images
    model.add(Conv2D(32, 4, strides=2, padding='same', input_shape=(64, 64, 3)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    ## convolutional block 2 - 32x32
    model.add(Conv2D(64, 4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    ## convolutional block 3 - 16x16
    model.add(Conv2D(128, 4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    ## convolutional block 4 - 8x8
    model.add(Conv2D(256, 4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    ## flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()

print(img.shape)
discriminator.predict(img)

# 4. Contruct Training Loop

## adam is going to be the optimiser for both
from keras.optimizers import Adam
## binary cross entropy is going to be the loss for both
from keras.losses import BinaryCrossentropy

g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

## importing the base model class to subclass our training step
from keras.models import Model

class NostalgiaGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        ## pass throguh args and kwargs to base class
        super().__init__(*args, **kwargs)

        ## create attributes for the gen and disc
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        ## compile with base class
        super().compile(*args, **kwargs)

        ## create attributes for the optimisers and losses
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        real_images = batch
        batch_size = tf.shape(real_images)[0]

        fake_images = self.generator(tf.random.normal((batch_size, 128)), training=False)

        # Real labels: smooth between 0.8 and 1.0
        real_labels = tf.random.uniform((batch_size, 1), 0.8, 1.0)

        # Fake labels: smooth between 0.0 and 0.2
        fake_labels = tf.random.uniform((batch_size, 1), 0.0, 0.2)

        # Occasionally flip labels (e.g., 10%)
        flip_mask = tf.random.uniform((batch_size, 1)) < 0.1
        real_labels = tf.where(flip_mask, fake_labels, real_labels)
        fake_labels = tf.where(flip_mask, real_labels, fake_labels)

        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)

            d_loss_real = self.d_loss(real_labels, yhat_real)
            d_loss_fake = self.d_loss(fake_labels, yhat_fake)
            total_d_loss = d_loss_real + d_loss_fake

        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            gen_images = self.generator(tf.random.normal((batch_size, 128)), training=True)
            predicted_labels = self.discriminator(gen_images, training=False)

            # Generator tries to fool discriminator into thinking fakes are real (labels=1)
            trick_labels = tf.ones_like(predicted_labels)
            total_g_loss = self.g_loss(trick_labels, predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {
            'd_loss': total_d_loss,
            'g_loss': total_g_loss
        }


## create instance of subclassed model
nostalgiagan = NostalgiaGAN(generator, discriminator)

## compile the model
nostalgiagan.compile(g_opt, d_opt, g_loss, d_loss)

## from keras.preprocessing.image import array_to_img
from keras.utils.image_utils import array_to_img
from keras.callbacks import Callback

## make sure the images directory exists
os.makedirs('images', exist_ok=True)

from keras.callbacks import ModelCheckpoint

## make sure the directory exists
os.makedirs('checkpoints', exist_ok=True)

# save generator weights every 10 epochs
checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints/generator_epoch_{epoch:03d}.h5',
    save_weights_only=True,
    save_freq=500,
    verbose=1
)

## train
hist = nostalgiagan.fit(ds, epochs=1000, callbacks=[checkpoint_callback])


## review performance
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='Discriminator Loss')
plt.plot(hist.history['g_loss'], label='Generator Loss')
plt.legend()
plt.show()


# 5. Test Out the Generator

num_images = 5
random_latent_vectors = tf.random.normal((num_images, 128))
generated_images = generator(random_latent_vectors)
generated_images = (generated_images + 1) / 2.0

fig, ax = plt.subplots(1, num_images, figsize=(num_images * 4, 4))

for i in range(num_images):
    ax[i].imshow(generated_images[i])
    ax[i].axis('off')

plt.show()
