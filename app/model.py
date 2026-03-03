"""
Model Architecture — TF/Keras neural network for classifying
hand landmarks into sign language gestures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_landmark_model(
    num_classes: int,
    input_dim: int = 63,
    hidden_units: list = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Dense neural network for landmark-based classification.

    Args:
        num_classes:   Number of sign language classes/labels.
        input_dim:     21 landmarks x 3 coords = 63.
        hidden_units:  Hidden layer sizes. Default [128, 64, 32].
        dropout_rate:  Dropout probability.
        learning_rate: Adam optimizer LR.
    Returns:
        Compiled Keras model.
    """
    if hidden_units is None:
        hidden_units = [128, 64, 32]

    inputs = keras.Input(shape=(input_dim,), name="landmarks_input")
    x = layers.BatchNormalization()(inputs)

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units, activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="sign_language_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_sequence_model(
    num_classes: int,
    seq_length: int = 30,
    num_features: int = 63,
    lstm_units: list = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    LSTM model for sequence-based sign recognition (dynamic gestures).
    """
    if lstm_units is None:
        lstm_units = [64, 32]

    inputs = keras.Input(shape=(seq_length, num_features), name="sequence_input")
    x = inputs

    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = layers.LSTM(units, return_sequences=return_seq, name=f"lstm_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"lstm_drop_{i}")(x)

    x = layers.Dense(64, activation="relu", name="fc_1")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="sign_sequence_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
