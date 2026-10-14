"""model.py - build the classifier (custom CNN or MobileNetV2 transfer learning)."""
from .config import IMG_SIZE


def build_model(transfer=False):
    import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
    from tensorflow.keras import layers, models  # pyright: ignore[reportMissingModuleSource]

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.RandomFlip("horizontal")(inp)          # data augmentation
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    if transfer:
        x = layers.Concatenate()([x, x, x])            # gray -> 3 channels
        try:
            base = tf.keras.applications.MobileNetV2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
        except Exception:
            print("   (ImageNet weights unavailable offline -> MobileNetV2 from scratch)")
            base = tf.keras.applications.MobileNetV2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None)
        base.trainable = False                          # freeze pretrained layers
        x = base(x * 2.0 - 1.0, training=False)         # MobileNet expects [-1, 1]
        x = layers.GlobalAveragePooling2D()(x)
    else:
        for filters in (32, 64, 128):                   # feature extraction blocks
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)      # P(PNEUMONIA)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
