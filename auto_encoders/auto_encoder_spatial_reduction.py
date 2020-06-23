import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers
input_shape = 100
latent_space_dim = 5
encoder_input = keras.Input(shape = (100), name ="variable")
en_x = layers.Dense(120, activation= "softplus")(encoder_input)
en_x = layers.Dense(60, activation="softplus")(en_x)
en_x = layers.Dense(30, activation="softplus")(en_x)
encoder_output = layers.Dense(5)(en_x)
encoder = keras.Model(encoder_input, encoder_output, name ="encoder")
encoder.summary()

decoder_input = keras.Input(shape = (5), name ="Latent_space")
de_x = layers.Dense(30, activation= "softplus")(decoder_input)
de_x = layers.Dense(60, activation="softplus")(de_x)
de_x = layers.Dense(120, activation="softplus")(de_x)
decoder_output = layers.Dense(input_shape)(de_x)
decoder = keras.Model(decoder_input, decoder_output, name ="decoder")
print(decoder.layers[1].get_weights())
decoder.summary()


auto_input = keras.Input(input_shape, name = "auto_input")
encoded = encoder(auto_input)
decoded = decoder(encoded)

auto_encoder = keras.Model(auto_input, decoded, name = "autoencoder")
auto_encoder.summary()
