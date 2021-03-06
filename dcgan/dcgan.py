import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
noise_dim = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 256
num_examples_to_generate = 16
EPOCHS = 1


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
 
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
 
    model.add(Flatten())
    model.add(Dense(1))
 
    return model


def test(generator, discriminator):
    noise = tf.random.normal([1, 100])
    print(noise)
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    decision = discriminator(generated_image)
    print(decision)


def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step_discriminator(generator,discriminator,images,noise):
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss


@tf.function
def train_step_generator(generator, discriminator, noise):
    with tf.GradientTape() as gen_tape :
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss


def generate_and_save_images(generator, epoch, test_input):
    predictions = generator(test_input, training=False)
 
    fig = plt.figure(figsize=(4,4))
 
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
 
    plt.savefig('./output/dcgan_image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def save(model, model_name):
    model_path = "saved_model/%s.json" % model_name
    weights_path = "saved_model/%s_weights.hdf5" % model_name
    options = {"file_arch": model_path,
               "file_weight": weights_path}
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])


def train(generator, discriminator, dataset, epochs):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    for epoch in range(epochs):
        print("epoch %d" % epoch)
        for i,image_batch in enumerate(dataset):
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            d = train_step_discriminator(generator, discriminator, image_batch, noise)
            g = train_step_generator(generator, discriminator, noise)
            print("batch %d, gen_loss %f,disc_loss %f" % (i, g.numpy(), d.numpy()))

    generate_and_save_images(generator, epochs, seed)
    save(generator,"dcgan_generator")
    save(discriminator, "dcgan_discriminator")


if __name__ == '__main__':
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    gen = make_generator_model()
    disc = make_discriminator_model()
    # test(gen,disc)
    train(gen, disc, train_dataset, EPOCHS)
