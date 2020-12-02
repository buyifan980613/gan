import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
noise_dim = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 256
num_examples_to_generate = 16
EPOCHS = 5


def cgan_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    noise = keras.Input(shape=(100,))
    label = keras.Input(shape=(1,))
    label_embedding = layers.Flatten()(layers.Embedding(10,100)(label))
    model_input = layers.multiply([noise,label_embedding])
    img = model(model_input)
    return Model([noise,label],img)


def cgan_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    img = keras.Input(shape=(28,28,1))
    label = keras.Input(shape=(1,))

    label_embedding = layers.Flatten()(layers.Embedding(10, np.prod((28,28,1)))(label))
    flat_img = layers.Flatten()(img)

    model_input = layers.multiply([flat_img, label_embedding])
    model_input = tf.reshape(model_input,[-1,28,28,1])
    validity = model(model_input)
    return Model([img,label],validity)


def test(generator, discriminator):
    noise = np.random.normal(0, 1, (1, 100))
    label = np.array([[3]])
    print(noise, label)
    generated_image = generator([noise, label], training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    decision = discriminator([generated_image, label], training=False)
    print(decision)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step_discriminator(generator, discriminator, images, noise, label):
    with tf.GradientTape() as disc_tape:
        generated_images = generator([noise,label], training=True)
        real_output = discriminator([images,label], training=True)
        fake_output = discriminator([generated_images,label], training=True)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss


@tf.function
def train_step_generator(generator, discriminator, noise, label):
    with tf.GradientTape() as gen_tape:
        generated_images = generator([noise,label], training=True)
        fake_output = discriminator([generated_images,label], training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss


def generate_and_save_images(generator, epoch, test_noise, test_label):
    predictions = generator([test_noise,test_label], training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.title(str(test_label[i][0]))
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./output/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    print(test_label)


def train(generator, discriminator, dataset, epochs):
    seed1 = tf.random.normal([num_examples_to_generate, noise_dim])  # noise
    seed2 = np.random.randint(10, size=(num_examples_to_generate, 1))  # label [[3],[4]]

    for epoch in range(epochs):
        print("epoch %d" % epoch)
        for i, image_batch in enumerate(dataset):
            image, label = image_batch
            noise = tf.random.normal([image.shape[0], noise_dim])
            d = train_step_discriminator(generator, discriminator, image, noise, label)
            g = train_step_generator(generator, discriminator, noise, label)
            print("batch %d, gen_loss %f,disc_loss %f" % (i, g.numpy(), d.numpy()))

    generate_and_save_images(generator, epochs, seed1, seed2)


if __name__ == '__main__':
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_labels = train_labels.reshape(-1, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    gen = cgan_generator()
    disc = cgan_discriminator()
    #test(gen, disc)
    train(gen, disc, train_dataset, EPOCHS)
