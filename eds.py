from tensorflow import keras
from keras import layers


def dense_encoder(latent_dim, variational=False):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(256, activation='leaky_relu'),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(64, activation='leaky_relu'),
    ], name='encoder')
    if variational:
        model.add(layers.Dense(latent_dim * 2))
    else:
        model.add(layers.Dense(latent_dim, activation='leaky_relu'))
    return model

def dense_decoder(latent_dim):
    return keras.Sequential([
        layers.InputLayer(input_shape=(latent_dim,)),
        layers.Dense(64, activation='leaky_relu'),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(256, activation='leaky_relu'),
        layers.Dense(28 * 28, activation='sigmoid'),
        layers.Reshape(target_shape=(28, 28, 1)),
    ], name='decoder')

def conv_encoder(latent_dim, variational=False):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Conv2D(filters=32, kernel_size=3, strides=2, 
            padding='same', activation='leaky_relu'),
        layers.Conv2D(filters=64, kernel_size=3, strides=2, 
            padding='same', activation='leaky_relu'),
        layers.Conv2D(filters=128, kernel_size=3, strides=2, 
            padding='same', activation='leaky_relu'),
        layers.Flatten(),
        layers.Dense(latent_dim * 4, activation='leaky_relu'),
    ], name='encoder')

    if variational:
        model.add(layers.Dense(latent_dim * 2))
    else:
        model.add(layers.Dense(latent_dim, activation='leaky_relu'))

    return model

def conv_decoder(latent_dim):
    return keras.Sequential([
        layers.InputLayer(input_shape=(latent_dim,)),
        layers.Dense(units=4*4*128, activation='leaky_relu'),
        layers.Reshape(target_shape=(4, 4, 128)),
        layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, 
            padding='same', activation='leaky_relu'),
        layers.Cropping2D(cropping=((0, 1), (0, 1))),
        layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
            padding='same', activation='leaky_relu'),
        layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
            padding='same', activation='leaky_relu'),
        layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1,
            padding='same', activation='sigmoid')
    ], name='decoder')

def cond_conv_encoder(latent_dim, num_classes):
    input_img = layers.Input((28, 28, 1))
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, 
        padding='same', activation='leaky_relu')(input_img)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, 
        padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, 
        padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    
    input_lbls = layers.Input(shape=(num_classes,), dtype='float32')
    x = layers.Concatenate()([x, input_lbls])
    x = layers.Dense(latent_dim * 4, activation='leaky_relu')(x)
    output = layers.Dense(latent_dim * 2)(x)

    return keras.models.Model([input_img, input_lbls], output, name='encoder')

def cond_conv_decoder(latent_dim, num_classes):
    z = layers.Input(shape=(latent_dim,))
    input_lbls = layers.Input(shape=(num_classes,), dtype='float32')
    x = layers.Concatenate()([z, input_lbls])

    x = layers.Dense(units=4*4*128, activation='leaky_relu')(x)
    x = layers.Reshape(target_shape=(4, 4, 128))(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, 
        padding='same', activation='leaky_relu')(x)
    x = layers.Cropping2D(cropping=((0, 1), (0, 1)))(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
        padding='same', activation='leaky_relu')(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
        padding='same', activation='leaky_relu')(x)
    output = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, 
        padding='same', activation='sigmoid')(x)

    return keras.models.Model([z, input_lbls], output, name='decoder')

def cond_dense_encoder(latent_dim, num_classes):
    input_img = layers.Input(shape=(28, 28, 1))
    flattened = layers.Flatten()(input_img)
    input_lbls = layers.Input(shape=(num_classes,), dtype='float32')
    input_layer = layers.Concatenate()([flattened, input_lbls])
    x = layers.Dense(256, activation='leaky_relu')(input_layer)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    output = layers.Dense(latent_dim * 2)(x)

    return keras.models.Model([input_img, input_lbls], output, name='encoder')

def cond_dense_decoder(latent_dim, num_classes):
    z = layers.Input(shape=(latent_dim,))
    input_lbls = layers.Input(shape=(num_classes,), dtype='float32')
    x = layers.Concatenate()([z, input_lbls])
    x = layers.Dense(64, activation='leaky_relu')(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(28 * 28, activation='sigmoid')(x)
    output = layers.Reshape(target_shape=(28, 28, 1))(x)

    return keras.models.Model([z, input_lbls], output, name='decoder')

