from keras.layers import Dense, Input, BatchNormalization, Conv2D, \
    LeakyReLU, Reshape, Conv2DTranspose, Flatten, Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
from keras.losses import mse
import numpy as np

import random
import librosa
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def rep_sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    eps = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var)*eps

latent_dim = 2000
original_dim = 512*256

#**********ENCODER STRUCTURE***************

inputs = Input(shape=(512, 256, 1))

encoded = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same')(inputs)
encoded = LeakyReLU(0.1)(encoded)
encoded = BatchNormalization()(encoded)


for fs in [64, 64, 128, 128, 128, 256, 256]:
    encoded = Conv2D(filters=fs, kernel_size=4, strides=(2, 2), padding='same')(encoded)
    encoded = LeakyReLU(0.1)(encoded)
    encoded = BatchNormalization()(encoded)


encoded = Conv2D(filters=256, kernel_size=4, strides=(2, 1), padding='same')(encoded)
encoded = LeakyReLU(0.1)(encoded)
encoded = BatchNormalization()(encoded)

encoded = Conv2D(filters=512, kernel_size=1, strides=(1, 1), padding='same')(encoded)
encoded = LeakyReLU(0.1)(encoded)
encoded = BatchNormalization()(encoded)
shape = K.int_shape(encoded) # Returns the shape of tensor or variable as a tuple of int or None entries.
print(shape) #int this structure is (None, 2, 1, 1024)

encoded = Flatten()(encoded)
z_mean = Dense(latent_dim, name='z_mean')(encoded)
z_log_var = Dense(latent_dim, name='z_log_var')(encoded)
z = Lambda(rep_sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='Encoder')

print("\n\tENCODER STRUCTURE")
encoder.summary()

#**********DECODER STRUCTURE***************

dec_inputs = Input(shape=(latent_dim, ))
decoded = Dense(shape[1] * shape[2] * shape[3])(dec_inputs) #shape_variable is the shape before
decoded = Reshape((shape[1], shape[2], shape[3]))(decoded)

decoded = Conv2DTranspose(filters=512, kernel_size=1, strides=(1, 1), padding='same')(decoded)
decoded = LeakyReLU(0.1)(decoded)
decoded = BatchNormalization()(decoded)

for fs in [256, 256, 128, 128, 128, 64, 64]:
    decoded = Conv2DTranspose(filters=fs, kernel_size=4, strides=(2, 2), padding='same')(decoded)
    decoded = LeakyReLU(0.1)(decoded)
    decoded = BatchNormalization()(decoded)


decoded = Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 2), padding='same')(decoded)
decoded = LeakyReLU(0.1)(decoded)
decoded = BatchNormalization()(decoded)

decoded = Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 1), padding='same')(decoded)
decoded = LeakyReLU(0.1)(decoded)
decoded = BatchNormalization()(decoded)

decoded = Conv2DTranspose(filters=1, kernel_size=1, strides=(1, 1), padding='same')(decoded)
decoder = Model(dec_inputs, decoded, name='decoder')

print("\n\tDECODER STRUCTURE")
decoder.summary()

#**********AUTOENCODER STRUCTURE***************
out = decoder(encoder(inputs)[2])
vae = Model(inputs, out, name='vae')
vae.summary()

# Loss definition
rec_loss = mse(K.flatten(inputs), K.flatten(out))
rec_loss *= original_dim
KL_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
KL_loss = K.sum(KL_loss, axis=-1)
KL_loss *= -0.5
vae_loss = K.mean(rec_loss + KL_loss)/original_dim

vae.add_loss(vae_loss)

my_opt = Adam(lr=1e-4)
vae.compile(optimizer=my_opt)

def my_generator(input_data, target, batch_size):
    # ----The generator for the train set----#
    while True:
        # Create empty arrays to contain batch of features and labels #
        batch_data = np.zeros((batch_size, dataset.shape[1], dataset.shape[2], 1))
        batch_target = np.zeros((batch_size, dataset.shape[1], dataset.shape[2], 1))
        for i in range(batch_size):
            # choose random index in features
            index = random.randrange(input_data.shape[0])
            batch_data[i] = input_data[index]
            batch_target[i] = target[index]
        yield batch_data, None

class SingleModelSaver(ModelCheckpoint):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):

        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only, mode=mode, period=period)

        self.base_model = base_model

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch + 1, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    if self.save_weights_only:
                        self.base_model.save_weights(filepath, overwrite=True)
                    else:
                        self.base_model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                              (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

def schedule_lr(epoch_index, lr):

    if epoch_index == 100 or epoch_index == 200:
        return 0.9*lr

    return lr

q = int(input("\n0 train | 1 predict | 2 generate in hypercube > "))

if q == 0:
    dataset = np.load('train_last_spec.npy')
    validation = np.load('valid_last_spec.npy')

    batch_size = 8

    train_gen = my_generator(dataset, dataset, batch_size)
    valid_gen = my_generator(validation, validation, batch_size)

    weights_saver = SingleModelSaver("weights_top.h5", vae, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=True, mode='auto', period=1)

    lr_scheduler = LearningRateScheduler(schedule_lr, verbose=1)

    vae.fit_generator(train_gen,
                      steps_per_epoch=len(dataset)//batch_size,
                      epochs=100,
                      shuffle=True,
                      validation_data=valid_gen,
                      validation_steps=len(validation)//batch_size,
                      callbacks=[weights_saver, lr_scheduler])

elif q == 1:
    from audio_util import *
    from rainbowgram import view_rainbowgram

    vae.load_weights("weights_top.h5")

    emb = np.random.random(size=(2000, )) #+ np.random.randint(100, 150, size=(2000, ))
    emb = np.expand_dims(emb, axis=0)

    test_audio = decoder.predict(emb)
    test_audio = np.squeeze(test_audio)
    test_audio = (test_audio - 1) * 120
    test_audio = 10 ** (test_audio / 20.0)
    test_audio = np.append(test_audio, np.zeros((test_audio.shape[0], 1)), axis=1)
    test_audio = np.append(test_audio, np.zeros((1, test_audio.shape[1])), axis=0)
    print(test_audio.shape)
    reconstruction = griffin_lim(test_audio, 0, 1024, 250, 400)
    reconstruction = np.squeeze(reconstruction / np.max(reconstruction))

    librosa.output.write_wav('rec.wav', reconstruction, sr=16000)
    view_rainbowgram('rec.wav')

elif q == 2:
    from audio_util import *
    from rainbowgram import save_rainbowgram

    vae.load_weights("weights_top.h5")

    num_audios = 20
    cube_side = 10
    for i in range(num_audios, ):
        test_z = np.random.uniform(-cube_side, -6, size=(latent_dim, ))
        test_z = np.expand_dims(test_z, axis=0)
        test_audio = decoder.predict(test_z)
        test_audio = np.squeeze(test_audio)

        test_audio = (test_audio - 1) * 120
        test_audio = 10 ** (test_audio / 20.0)
        test_audio = np.append(test_audio, np.zeros((test_audio.shape[0], 1)), axis=1)
        test_audio = np.append(test_audio, np.zeros((1, test_audio.shape[1])), axis=0)
        print(test_audio.shape)
        reconstruction = griffin_lim(test_audio, 0, 1024, 250, 400)
        reconstruction = np.squeeze(reconstruction / np.max(reconstruction))

        librosa.output.write_wav('hypercube_side4/rec00357_'+str(i)+'.wav', reconstruction, sr=16000)
        save_rainbowgram('hypercube_side4/rec00357_'+str(i)+'.wav', 'hypercube_side4/rainb00357_'+str(i)+'.png')