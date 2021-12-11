import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import glob

import models
import eds
from math import ceil

def create_gif(anim_file, folder='./', exclude=[]):
    with imageio.get_writer(folder + anim_file, mode='I') as writer:
        filenames = glob.glob(folder + '*.png')
        filenames = [i for i in filenames if not i in exclude]
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def save_sample(sample, path):
    fig = plt.figure(figsize=(8, 8))
    n = int(ceil(sample.shape[0] ** (1/2)))
    for i in range(sample.shape[0]):
      plt.subplot(n, n, i + 1)
      plt.imshow(sample[i, :, :, 0], cmap='gray')
      plt.axis('off')

    plt.savefig(path)
    plt.close(fig)


class UpdateImagesCb(keras.callbacks.Callback):
    def __init__(self, model, samples, gen_from, origs=None, folder='./'):
        for path in ['reconstructed', 'generated']:
            Path(folder + path).mkdir(parents=True, exist_ok=True)

        self.gen_from = gen_from
        self.folder = folder
        self.model = model
        self.samples = samples
        if origs is None:
            origs = samples
        save_sample(origs, folder + 'orig.png')

    def on_epoch_end(self, epoch, logs=None):
        reconstructed = self.model.predict(self.samples)
        path = self.folder + 'reconstructed/{:02d}.png'.format(epoch+1)
        save_sample(reconstructed, path)

        path = self.folder + 'generated/{:02d}.png'.format(epoch+1)
        generated = self.model.decode(self.gen_from)
        save_sample(generated, path)

    def on_train_end(self, logs=None):
        create_gif(anim_file='generated.gif', 
            folder=folder + 'generated/')
        create_gif(anim_file='reconstructed.gif', 
            folder=folder + 'reconstructed/', exclude='orig.png')
        

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    # def on_epoch_end(self, epoch, logs={}):
    #     self.losses.append(logs.get('loss'))

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


optimizer = keras.optimizers.Adam()
batch_size = 128
epochs = 10
latent_dim = 2
folder = 'asdf/'
cp_path = folder + 'ckpt'
ckpt_cb = keras.callbacks.ModelCheckpoint(
    filepath = cp_path,
    monitor='loss',
    save_best_only=True,
    save_weights_only=True,
    save_freq="epoch",
)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
dataset = np.concatenate([x_train, x_test])
dataset = np.expand_dims(dataset, -1).astype("float32") / 255

y_dataset = np.concatenate([y_train, y_test], axis=0)
y_cat = keras.utils.to_categorical(y_dataset).astype(np.float32)
num_classes = y_cat.shape[1]


encoder = eds.cond_conv_encoder(latent_dim, num_classes)
decoder = eds.cond_conv_decoder(latent_dim, num_classes)
encoder.summary()
decoder.summary()

ae = models.CVAE(encoder, decoder, latent_dim, optimizer)
ae.compile(optimizer=optimizer)

noise = np.random.normal(size=(num_classes ** 2, latent_dim))
cats = np.tile(np.eye(num_classes), num_classes).T
uicb = UpdateImagesCb(ae, samples=(dataset[:16], y_cat[:16]), 
    origs=dataset[:16], gen_from=(noise, cats), folder=folder)

hist = LossHistory()
ae.fit((dataset, y_cat), epochs=epochs, 
    batch_size=batch_size, callbacks=[uicb, hist, ckpt_cb])

fig = plt.figure()
plt.plot(hist.losses)
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig(folder + 'loss_plot.png')
plt.close(fig)

