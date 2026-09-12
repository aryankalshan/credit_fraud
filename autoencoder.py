"""Stage 1: learn normal behavior, then extract representations for every class."""
import numpy as np
import pandas as pd
import tensorflow as tf


def train_autoencoder(train, train_labels, valid, valid_labels, latent_dim=8,
                      epochs=30, batch_size=512, seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    if not 0 < latent_dim < train.shape[1]:
        raise ValueError("The bottleneck must be smaller than the input dimension.")
    normal_train = train[np.asarray(train_labels) == 0]
    normal_valid = valid[np.asarray(valid_labels) == 0]
    inputs = tf.keras.Input(shape=(train.shape[1],), name="transactions")
    hidden = tf.keras.layers.Dense(32, activation="relu")(inputs)
    latent = tf.keras.layers.Dense(latent_dim, name="latent")(hidden)
    hidden = tf.keras.layers.Dense(32, activation="relu")(latent)
    outputs = tf.keras.layers.Dense(train.shape[1], name="reconstruction")(hidden)
    model = tf.keras.Model(inputs, outputs, name="normal_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    history = model.fit(
        normal_train, normal_train, validation_data=(normal_valid, normal_valid),
        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    encoder = tf.keras.Model(model.input, model.get_layer("latent").output)
    return model, encoder, pd.DataFrame(history.history)


def extract_features(model, encoder, scaled, index):
    latent = encoder.predict(scaled, batch_size=4096, verbose=0)
    reconstructed = model.predict(scaled, batch_size=4096, verbose=0)
    error = np.mean(np.square(scaled - reconstructed), axis=1)
    features = pd.DataFrame(latent, index=index,
                            columns=[f"ae_{i + 1}" for i in range(latent.shape[1])])
    features["ae_error"] = error
    assert len(features) == len(scaled) and np.isfinite(features.to_numpy()).all()
    return features
