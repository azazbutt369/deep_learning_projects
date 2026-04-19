This folder contains the base code for following CNN models:
- EfficientNet V2B3
- EfficientNet B4
- DenseNet 121
- XceptionNet
- Ensemble Model

For each CNN model, the following logic is applied:

``
base_model = <PretrainedModel>(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
``
