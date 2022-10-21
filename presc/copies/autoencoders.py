"""Convolutional Variational AutoEncoder class adapted from 'Variational
AutoEncoder' (2020/05/03) from fchollet (https://twitter.com/fchollet)."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    Flatten,
    Dense,
    Reshape,
    Conv2DTranspose,
)


class VAE(keras.Model):
    """Convolutional Variational AutoEncoder (VAE).

    This class takes encoder and decoder models and defines the complete
    variational autoencoder architecture as a keras model with a custom
    `train_step`.

    Parameters
    ----------
    encoder : Keras Model
        Encoder model for the variational autoencoder. It should include the
        sampling variational layer.
    decoder : Keras Model
        Decoder model for the variational autoencoder.
    """

    def __init__(self, encoder, decoder, **kwargs):
        """Constructor for instance of the Convolutional Variational AE class."""
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """metrics.

        Returns
        -------
        list of Keras Mean metrics
            List of the three metrics that are tracked at every step of the
            autoencoder training process: the total loss, and the two components
            the reconstruction loss and the KL-divergence loss.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """Custom train step for the Convolutional Variational Autoencoder.

        The variational autoencoder uses a custom total loss which consists of
        two terms, one is the reconstruction error and the other is the
        Kullback–Leibler divergence (or relative entropy) loss. The
        KL-divergence measures how similar two probability distributions are, so
        the variational autoencoder uses KL-divergence as part of its loss
        function in order to minimize the difference between a supposed
        distribution and the original distribution of the dataset.

        Parameters
        ----------
        data : ??
            Data used for the train step.

        Returns
        -------
        dict of floats
            A dictionary with the total loss, the reconstruction loss, and the
            KL-divergence loss.
        """
        if isinstance(data, tuple):
            data = data[0]

        # tf.GradientTape records operations for later retrieval for automatic
        # differentiation.
        with tf.GradientTape() as tape:
            # Compute output of the autoencoder with given data
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Compute loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        # Computes gradients using operations recorded in context of this tape.
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Updates weights using the previously recorded gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Updates values in loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class Sampling(Layer):
    """Creates the variational AE sampling layer with a Gaussian distribution."

    This bottleneck layer is used at the end of the encoder part of the
    variational autoencoder to build the sampling layer of the autoencoder.
    Here each dimension in the latent space will be represented by a Gaussian
    distribution with a mean and a variance. This function uses the mean and the
    log variance of a Gaussian distribution to sample z, the vector in latent
    space encoding the image representation.
    """

    def call(self, inputs):
        """Draws a sample of latent space vector z using Gaussian distributions.

        Parameters
        ----------
        inputs : tuple
            The mean and the log variance of the Gaussian distribution of z, the
            vector that represents the original sample in the latent space.

        Returns
        -------
        vector?/tensor?
            Returns a sample of the vector z drawn from a the probabilistic
            representation in Gaussian distributions of the latent space.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ImageVAE:
    """Convolutional Variational Autoencoder for image samples.

    This class implements a variational autoencoder where the encoder and the
    decoder are convolutional networks, which are useful for problems with
    images.

    Parameters
    ----------
    input_shape : tuple of int
        Three integers describing the shape of the input images. That is, the
        number of pixels in each of the two dimensions, and the number of color
        channels (for black and white images is 1 channel, and for RGB images
        is 3 channels). The number of pixels of the images need to be divisible
        by 4 in both dimensions. Default shape value is (28, 28, 1).
    latent_dim : int
        The number of latent dimensions.
    optimizer : Keras optimizer
        The optimizer to use. There are multiple options.

        One example is an Adam optimizer, which is a stochastic gradient descent
        method that is based on adaptive estimation of first-order and
        second-order moments: `keras.optimizers.Adam()`. This optimizer is
        computationally efficient, has little memory requirements, is invariant
        to diagonal rescaling of gradients, and is well suited for problems that
        are large in terms of data or parameters.

        Another example of optimizer is RMSprop, which mantains a moving
        (discounted) average of the square of gradients, and divides the
        gradient by the root of this average: `tf.keras.optimizers.RMSprop()`.
        This implementation of RMSprop uses plain momentum, not Nesterov
        momentum. And the centered version (centered=True) additionally
        maintains a moving average of the gradients, and uses that average to
        estimate the variance (it helps with training, but is slightly more
        expensive in terms of computation and memory).
    """

    def __init__(self, input_shape=(28, 28, 1), latent_dim=2, optimizer="rmsprop"):
        """Constructor for the Convolutional Variational Autoencoder."""
        # Check that image dimensions are divisible by 4
        if (np.array(input_shape[:2]) % 4).any() != 0:
            raise Exception("Sorry, image dimensions must be divisible by 4.")

        # Compute dimensions for the decoder
        dimensions_decoder = ((np.array(input_shape[:2]) / 2) / 2).astype(int)

        # Define encoder
        encoder_inputs = Input(shape=input_shape)

        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)

        # Bottleneck layer
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)

        z = Sampling()([z_mean, z_log_var])

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Define Decoder Architecture
        latent_inputs = keras.Input(shape=(latent_dim,))

        x = Dense(
            dimensions_decoder[0] * dimensions_decoder[1] * 64, activation="relu"
        )(latent_inputs)
        x = Reshape((dimensions_decoder[0], dimensions_decoder[1], 64))(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

        decoder_outputs = Conv2DTranspose(
            input_shape[2], 3, activation="sigmoid", padding="same"
        )(x)

        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        # compile and instantiate
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=optimizer)

    def fit(self, images, **kwargs_fit):
        """Train the Convolutional Variational Autoencoder model.

        Parameters
        ----------
        images : ??
            Image dataset to use for the training.
        epochs : int
            Number of epochs to train the model for.
        batch_size : int
            Batch size of the samples to use for the training.

        Returns
        -------
        self
            The fitted convolutional variational autoencoder.
        """
        self.vae.fit(images, **kwargs_fit)
        return self
