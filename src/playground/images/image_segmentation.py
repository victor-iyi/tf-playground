# Copyright 2022 Victor I. Afolabi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix


dataset, info = tfds.load(
    'oxford_iiit_pet:3.*.*', data_dir='../../data/', with_info=True
)

OUTPUT_CLASSES, VAL_SUBSPLITS = 3, 5
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = 128, 128, 3

EPOCHS, LEARNING_RATE = 20, 1e-4
BATCH_SIZE, BUFFER_SIZE = 64, 1000

TRAIN_LENGTH = info.splits['train'].num_examples
VAL_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        # Both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(
            mode='horizontal', seed=seed
        )
        self.augment_labels = tf.keras.layers.RandomFlip(
            mode='horizontal', seed=seed
        )

    def call(
        self,
        inputs: tf.Tensor,
        labels: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)

        return inputs, labels


def load_data(train: bool = True) -> tf.data.Dataset:
    images = dataset['train' if train else 'test'].map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    if train:
        batches = (
            images
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .repeat()
            .map(Augment(), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
    else:
        batches = images.batch(BATCH_SIZE)
    return batches


def normalize(
    input_image: tf.Tensor,
    input_mask: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(
    datapoint: dict[str, tf.Tensor]
) -> tuple[tf.Tensor, tf.Tensor]:
    input_image = tf.image.resize(datapoint['image'],
                                  (IMG_HEIGHT, IMG_WIDTH))
    input_mask = tf.image.resize(datapoint['segmentation_mask'],
                                 (IMG_HEIGHT, IMG_WIDTH))
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def display(display_lists: list[tf.TensorArray]) -> None:
    plt.figure(figsize=(5, 5))

    title = ['Input Image', 'True Mask', 'Predicted Mask']
    n_display = len(display_lists)

    for i, display_list in enumerate(display_lists):
        plt.subplot(1, n_display, i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list))
        plt.axis('off')

    plt.show()


def create_mask(pred_mask: tf.Tensor) -> tf.TensorArray:
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def encoder_stack(
    input_shape: tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)
) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False
    )

    # Use the activations of these layers.
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output
                          for name in layer_names]

    # Create the feature extraction model.
    down_stack = tf.keras.Model(inputs=base_model.input,
                                outputs=base_model_outputs)
    down_stack.trainable = False

    return down_stack


def decoder_stack() -> list[tf.keras.Model]:
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    return up_stack


def unet_model(
    output_channels: int = OUTPUT_CLASSES,
    input_shape: tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling through the model.
    down_stack = encoder_stack(input_shape)
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections.
    up_stack = decoder_stack()
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model.
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same', activation='softmax'
    )  # 64x64 -> 128x128
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def main() -> int:
    train_batches = load_data(train=True)
    # test_batches = load_data(train=False)

    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask])

    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # tf.keras.utils.plot_model(model, show_shapes=True)
    model.summary()

    def show_predictions(
        dataset: tf.dataset.Dataset | None = None,
        num: int = 1
    ) -> None:
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                display([image[0], mask[0], create_mask(pred_mask)])
        else:
            display([sample_image, sample_mask,
                    create_mask(model.predict(sample_image[tf.newaxis, ...]))])

    show_predictions()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
