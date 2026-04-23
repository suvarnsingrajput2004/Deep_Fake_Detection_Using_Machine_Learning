from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, EfficientNetB0, Xception


# ---------------------------------------------------------------------------
# Custom Loss: Focal Loss (handles class imbalance better than BCE)
# ---------------------------------------------------------------------------
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss — down-weights easy examples so the model focuses on hard ones."""

    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply label smoothing
        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(focal_weight * bce)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha, "label_smoothing": self.label_smoothing})
        return config


# ---------------------------------------------------------------------------
# Helper: common classification head
# ---------------------------------------------------------------------------
def _add_classification_head(x, dropout_1=0.5, dropout_2=0.3):
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_2)(x)
    return layers.Dense(1, activation="sigmoid")(x)


def _compile(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 1: MobileNetV2 (lightweight CNN)
# ---------------------------------------------------------------------------
def build_cnn(input_shape=(224, 224, 3)):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    output = _add_classification_head(x)

    model = models.Model(inputs=base.input, outputs=output)
    return _compile(model)


# ---------------------------------------------------------------------------
# Model 2: ResNet50V2
# ---------------------------------------------------------------------------
def build_resnet(input_shape=(224, 224, 3)):
    base = ResNet50V2(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    output = _add_classification_head(x)

    model = models.Model(inputs=base.input, outputs=output)
    return _compile(model)


# ---------------------------------------------------------------------------
# Model 3: EfficientNetB0
# ---------------------------------------------------------------------------
def build_efficientnet(input_shape=(224, 224, 3)):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    output = _add_classification_head(x)

    model = models.Model(inputs=base.input, outputs=output)
    return _compile(model)


# ---------------------------------------------------------------------------
# Model 4: XceptionNet (SOTA for deepfake detection — FaceForensics++ paper)
# ---------------------------------------------------------------------------
def build_xception(input_shape=(224, 224, 3)):
    """XceptionNet — the gold standard architecture for face-swap deepfake detection.
    
    The original FaceForensics++ paper demonstrated that Xception outperforms
    all other architectures for detecting face manipulations due to its
    depthwise separable convolutions capturing fine-grained texture artifacts.
    """
    base = Xception(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers[:-40]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    output = _add_classification_head(x, dropout_1=0.5, dropout_2=0.3)

    model = models.Model(inputs=base.input, outputs=output)
    return _compile(model, lr=5e-5)


# ---------------------------------------------------------------------------
# Model 5: CNN-LSTM (spatial sequence analysis)
# ---------------------------------------------------------------------------
def build_lstm(input_shape=(224, 224, 3)):
    """Spatial LSTM model that processes CNN features as a sequence."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    # Reshape spatial dimensions (7x7) into a sequence of length 49
    shape = x.shape
    x = layers.Reshape((shape[1] * shape[2], shape[3]))(x)

    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base.input, outputs=output)
    return _compile(model)


# ---------------------------------------------------------------------------
# Model 6: GAN Discriminator
# ---------------------------------------------------------------------------
def build_discriminator(input_shape=(224, 224, 3)):
    """A standard GAN Discriminator architecture."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="discriminator")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
        loss=FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 7: Vision Transformer (ViT)
# ---------------------------------------------------------------------------
def build_vit(input_shape=(224, 224, 3)):
    """A minimal Vision Transformer (ViT) architecture."""
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_layers = 2

    inputs = layers.Input(shape=input_shape)

    # Extract patches and project
    x = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    x = layers.Reshape((num_patches, projection_dim))(x)

    # Add position embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + position_embedding

    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=tf.nn.gelu)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="vit")
    return _compile(model)
