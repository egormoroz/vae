import tensorflow as tf
from tensorflow import keras


class AE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def call(self, x):
        return self.decode(self.encode(x))

    def sample(self, noise=None):
        if noise is None:
            noise = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(noise)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim, optimizer):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.optimizer = optimizer

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps)

    def encode(self, data):
        encoded = self.encoder(data)
        mean, logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, data):
        return self.decoder(data)

    def handle_losses(self, losses, tape):
        total_loss, kl_loss, reconstruction_loss = losses
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(data)
            z = reparameterize(mean, logvar)
            reconstruction = self.decode(z)
            losses = vae_losses(mean, logvar, data, reconstruction)

        return self.handle_losses(losses, tape)
        
    def call(self, data):
        mean, logvar = self.encode(data)
        z = reparameterize(mean, logvar)
        return self.decode(z)


class CVAE(VAE):
    @tf.function
    def train_step(self, data):
        data = data[0]
        x, lbls = data
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(data)
            z = reparameterize(mean, logvar)
            reconstruction = self.decode((z, lbls))
            losses = vae_losses(mean, logvar, x, reconstruction)

        return self.handle_losses(losses, tape)

    def call(self, data):
        _, lbls = data
        z_mean, z_log_var = self.encode(data)
        z = reparameterize(z_mean, z_log_var)
        return self.decode((z, lbls))


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=tf.shape(mean))
    return eps * tf.exp(logvar * 0.5) + mean

def vae_losses(z_mean, z_log_var, data, reconstruction):
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
        )
    )
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    total_loss = reconstruction_loss + kl_loss

    return total_loss, kl_loss, reconstruction_loss

