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

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


dataset, info = tfds.load(
    'oxford_iiit_pet:3.*.*', data_dir='../../data/', with_info=True
)
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE, BUFFER_SIZE = 64, 1000
TRAIN_LENGTH = info.splits['train'].num_examples
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


def main() -> int:
    train_batches = load_data(train=True)
    # test_batches = load_data(train=False)

    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
