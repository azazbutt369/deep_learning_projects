# src/training/trainer.py

from tensorflow.keras.optimizers import Adam

from .callbacks import get_callbacks
from .losses import get_loss

def compile_model(model):
    """
    Compiling configuration from notebook
    """

    model.compile(
        optimizer=Adam(),
        loss=get_loss(),
        metrics=['accuracy']
    )

    return model


def train_model(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    epochs=20,
    batch_size=32
):
    """
    Training logic
    """

    callbacks = get_callbacks()

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history
