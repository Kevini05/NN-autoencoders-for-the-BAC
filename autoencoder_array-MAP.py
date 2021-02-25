# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf

import utils
import utils_ML

import numpy as np
import matplotlib.pyplot as plt
from mish import Mish as mish

def max_pyx(y,codebook, e0, e1):
  """returns the Maximum Likelihood index from a
  received codeword y, compared with all elements in the codebook """
  Pyx_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  two = tf.cast(2, tf.float32)
  for code in codebook:
    d11 = K.sum(code * y)
    d10 = K.sum(tf.math.floormod(code + 1, two) * y)
    d01 = K.sum(code * tf.math.floormod(y + 1, two))
    f = d01 * K.log(e0 / (1 - e0)) + d10 * K.log(e1 / (1 - e0)) + d11 * K.log((1 - e1) / (1 - e0))
    Pyx_list = Pyx_list.write(Pyx_list.size(), f)
  Pyx_list = Pyx_list.stack()
  return K.argmax(Pyx_list)

def map(U_sequences,e0,e1,k):
  """returns the Mean Squared Error between the transmitted messages (u_true)
    and the estimated sequences (u_pred), both of size k.
    This estimation is made by the function "max_pyx" which computes the Maximum likelihood
    :param U_sequences: represent all possible sequences of size k.
    TRICK: to create a loss function with other parameters than label and prediction,
      you can create a function like 'map' which receives any arguments
      and inside this one create a second function that receives the label and prediction"""
  U_sequences = tf.cast(U_sequences, tf.float32)
  def map_loss_function(u_true, u_pred):

    u_true = tf.cast(u_true, tf.float32)
    encoder_bits = tf.split(u_pred, num_or_size_splits = 2, axis = 1)[0]
    noisy_bits = tf.split(u_pred, num_or_size_splits=2, axis=1)[1]
    codebook = tf.cast(encoder_bits[0:2**k], tf.float32)
    codebook = K.round(codebook)

    u_k_hat = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for y in noisy_bits:
      idx_map = max_pyx(y, codebook, e0, e1)
      u_k_hat = u_k_hat.write(u_k_hat.size(), tf.gather(U_sequences, idx_map))

    u_k_hat = u_k_hat.stack()

    return tf.keras.losses.mean_squared_error(u_k_hat, u_true) #K.mean(U_label-Y_all) #

  return map_loss_function

####################################################################################################
########### Neural Network Generator ###################

### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape=k, name='input_encoder')
  x = Dense(units=256, activation=activation, kernel_initializer=initializer)(inputs_encoder)
  x = BatchNormalization()(x)
  x = Dense(units=128, activation=activation, kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  outputs_encoder = Dense(units=N, activation='sigmoid', kernel_initializer=initializer)(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder')
  return model_enc

### Meta model Layers definitions
def meta_model_generator(k,model_enc,round,epsilon_t):
  inputs_meta = keras.Input(shape=k, name='input_meta')
  encoded_bits = model_enc(inputs=inputs_meta)
  if round:
    x = Lambda(utils_ML.gradient_stopper, name='rounding_layer')(encoded_bits)
  else:
    x = encoded_bits
  noisy_bits = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_t, 'epsilon_1_max': train_epsilon_1}, name='noise_layer')(x)
  outLayer = K.concatenate([x, noisy_bits], axis=-1)
  ### Model Build
  meta_model = keras.Model(inputs=inputs_meta, outputs=outLayer, name = 'meta_model')
  return meta_model

# Command line Parameters
N = int(sys.argv[1])
k = int(sys.argv[2])
nb_pkts = int(sys.argv[3])
length_training = sys.argv[4]

# Select training and test length
if length_training == 'medium':
  rep = 128
  epoch_pretrain = 300
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 3000 if nb_pkts < 3000 else nb_pkts
elif length_training == 'bug':
  rep = 2
  epoch_pretrain = 10
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 5)),axis=0)
  verbose = 2
  nb_pkts = 10
elif length_training == 'long':
  rep = 256
  epoch_pretrain = 1000
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 0
  nb_pkts = 10000 if nb_pkts < 10000 else nb_pkts
else:
  rep = 128
  epoch_pretrain = 100
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2

e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
e1 = [t for t in e0 if t <= 0.5]

#Parameters
MAP_test = False
train_epsilon_1 = 0.0002       #useless for the BSC and epsilon_1 for the BAC
pretrain_epsilon = 0.1

#Training Data set
u_k = utils.symbols_generator(k)
U_k = np.tile(u_k,(rep,1))
In = utils.symbols_generator(N)[0:2**k] # List of outputs of NN
In = np.tile(In,(rep,1)).astype(float)

#Hyper parameters
batch_size = 256
initializer = tf.keras.initializers.Orthogonal()
loss = 'mse' #'categorical_crossentropy'  #'kl_divergence'          # 'mse'
activation = 'Mish' # Activation function for hidden layers

lr = 0.1
decay = 0.999
# reducing the learning rate every epoch
cbks = [LearningRateScheduler(lambda epoch: lr * decay ** (epoch // 1))]
optimizer = keras.optimizers.Nadam(lr=lr)
lr_metric = utils_ML.get_lr_metric(optimizer)

# Saved results recovery for plot them later
BER = utils.read_ber_file(N, k, 'BER')
BER = utils.saved_results(BER, N, k)

# # pretraining
print("----------------------------------Training ------------------------------------------")
model_encoder = encoder_generator(N,k)
meta_model = meta_model_generator(k,model_encoder, False, pretrain_epsilon)
### Compile our models
meta_model.compile(loss=map(u_k,pretrain_epsilon,train_epsilon_1,k), optimizer=optimizer)
### Fit the model
history = meta_model.fit(U_k, U_k, epochs=epoch_pretrain, verbose=verbose, shuffle=False, batch_size=batch_size, callbacks=cbks)

loss_values = history.history['loss']

C = np.round(model_encoder.predict(u_k)).astype('int')
BER['NN_Encoder-MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1, coded = True)


##########################################################################################################################
# Plot the loss function values
fig = plt.figure(figsize=(20,10))
title = f'N={N} k={k} {length_training} - NN Array-MAP'
plt.semilogy(loss_values  , alpha=0.8 , color='brown',linewidth=0.15, label='Loss')
# Plot the loss function values passed through a filter,
filter_size = 100
plt.semilogy(utils_ML.smooth(loss_values,filter_size)[filter_size-1:], color='brown')

plt.title(f'{title} - Training results vs No. epoch - {nb_pkts} pkts')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="best")
plt.grid()
fig.savefig(f"./figures/LOSS {title}")
#####################################################
# BER plotting
utils.plot_BAC(f'BER {title}', BER, k / N)

print(title)
# plt.show()

# \Python3\python.exe autoencoder_array-MAP.py 8 4 1000 medium