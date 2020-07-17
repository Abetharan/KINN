import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers
from random import gauss
import h5p
import os 

def autoencoder(input_shape, latent_output_space):
    encoder_input = keras.Input(shape = (input_shape), name ="variable")
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
    #print(decoder.layers[1].get_weights())
    decoder.summary()


    auto_input = keras.Input(input_shape, name = "auto_input")
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto_encoder = keras.Model(auto_input, decoded, name = "autoencoder")
    auto_encoder.summary()
    return auto_encoder


def get_compiled_model(input_shape, latent_output_space, learning_rate):
    model = autoencoder(input_shape, latent_output_space)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanSquaredError(),
        # List of metrics to monitor
        metrics=[keras.metrics.MeanSquaredError()],
    )
    return model

def make_or_restore_model(input_shape, latent_output_space, checkpoint_dir, learning_rate):
  # Either restore the latest model, or create a fresh one
  # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(input_shape, latent_output_space, learning_rate)


#Changing learning rate
#initial_learning_rate = 0.1
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
#)
#optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

checkpoint_dir = "/content/drive/My Drive/ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def normalise(x):
  mean = np.mean(x)
  std = np.std(x)
  normed = (x - mean) / std
  return mean, std, normed


input_shape = 99
latent_output_space = 3
x_train = np.array(k["Train/GradTe/"]['data'])#.reshape(8000,99)
mean, std, x_train = normalise(x_train)
x_train = x_train.reshape(8000,99)
#x_test = normalise(x).reshape(2000,99)
#model = autoencoder(input_shape, latent_output_space)
#model.compile(
#    optimizer=keras.optimizers.Adam(1e-3),  # Optimizer
#    # Loss function to minimize
#    loss=keras.losses.MeanSquaredError(),
#    # List of metrics to monitor
#    metrics=[keras.metrics.MeanSquaredError()],
#)

#model = make_or_restore_model(input_shape, latent_output_space, checkpoint_dir, learning_rate = 1e-3)

dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))

#Diagonistics
call_backs = [
  keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-4,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    ),
 keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath= checkpoint_dir + "/ckpt-loss={loss:.2f}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
 ),
  keras.callbacks.TensorBoard(
      log_dir= checkpoint_dir + "/full_path_to_your_logs",
      histogram_freq=0,  # How often to log histogram visualizations
      embeddings_freq=0,  # How often to log embedding visualizations
      update_freq="epoch",
  )  # How often to write logs (default: once per epoch)
]

history = model.fit(
    x_train,
    x_train,
    batch_size=64,
    epochs=200,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_split=0.15,
    callbacks=call_backs
)

history.history