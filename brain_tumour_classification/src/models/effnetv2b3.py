from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def build_effnetv2b3(input_shape=(150, 150, 3), num_classes=4):
    base_model = EfficientNetV2B3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model
