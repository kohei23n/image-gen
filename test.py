# 1. Importing Dependencies and Data
## bringing in tensorflow
import tensorflow as tf

## set memory growth for GPUs to avoid memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## bringing in tensorflow datasets for fashion mnist
import tensorflow_datasets as tfds
## bringing in matplotlib for visualisation
from matplotlib import pyplot as plt

## use the tensorflow datasets api to bring in the data source
ds = tfds.load('fashion_mnist', split='train')


# 2. Visualise Data and Build Dataset
## do some data transformation
import numpy as np

## setup connection aka iterator
dataiterator = ds.as_numpy_iterator()

## setup the subplot formatting
fig, ax = plt.subplots(ncols=4, figsize=(20,20))

## loop four times and get images
for idx in range(4):
    ## grab an image and label
    sample = dataiterator.next()
    ## plot the image using a specific subplot 
    ax[idx].imshow(np.squeeze(sample['image']))
    ## appending the image label as the plot title
    ax[idx].title.set_text(sample['label'])

## show the images
# plt.show()

## scale and return images only
def scale_images(data):
    image = data['image']
    return image / 255

## reload the dataset
ds = tfds.load('fashion_mnist', split='train')
## running the dataset through the scale_images preprocessing function
ds = ds.map(scale_images)
## cache the dataset for that batch
ds = ds.cache()
## shuffle it up
ds = ds.shuffle(60000)
## batch into 128 images per sample
ds = ds.batch(128)
## reduces the likelihood of bottlenecking
ds = ds.prefetch(64)


# 3. Build Neural Network

## bring in the sequential api for the generator and discriminatoir
from keras.models import Sequential
## bringing in the layers for the neural network
from keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

def build_generator():
    model = Sequential()
    
    ## takes in random values and reshapes it to 7x7x128
    ## beginnings of a generated image
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    ## upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    ## upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    ## convolutional block 1
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    ## convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    ## conv layer to get to one channel
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model

generator = build_generator()

## generate new fashion
img = generator.predict(np.random.randn(4, 128, 1))
print(img.shape)

## setup the subplot formatting
fig, ax = plt.subplots(ncols=4, figsize=(20,20))

## loop four times and get images
for idx, single_img in enumerate(img):
    ## plot the image using a specific subplot 
    ax[idx].imshow(np.squeeze(single_img))
    ## appending the image label as the plot title
    ax[idx].title.set_text(idx)

## show the images
# plt.show()

def build_discriminator():
    model = Sequential()

    ## convolutional block 1
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    ## convolutional block 2
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    ## convolutional block 3
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    ## convolutional block 4
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    ## flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
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

class FashionGAN(Model):
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
        ## get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        ## train the discriminator
        with tf.GradientTape() as d_tape:
            ## pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            ## create labels for real and fake images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            ## add some noise to the TRUE outputs
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = 0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            ## calcuate loss
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        ## apply background propagation -- nn learn
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        
        ## train the generator
        with tf.GradientTape() as g_tape:
            ## generate some new images
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            ## create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)

            ## calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ## apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {
            'd_loss': total_d_loss,
            'g_loss': total_g_loss
        }

## create instance of subclassed model
fashgan = FashionGAN(generator, discriminator)

## compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

## build calllback
import os
from keras.preprocessing.image import array_to_img
from keras.callbacks import Callback

class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal((self.num_img, self.latent_dim, 1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))

## train
hist = fashgan.fit(ds, epochs=20, callbacks=[ModelMonitor()])

## review performance
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='Discriminator Loss')
plt.plot(hist.history['g_loss'], label='Generator Loss')
plt.legend()
plt.show()


# 5. Test Out the Generator

imgs = generator.predict(tf.random.normal((16, 128, 1)))
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 20))
for r in range(4):
    for c in range(4):
        ax[r, c].imshow(imgs[(r+1)*(c+1)-1])
