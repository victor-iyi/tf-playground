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

EPOCHS, LEARNING_RATE = 5, 1e-3
BATCH_SIZE, BUFFER_SIZE = 32, 1000
N_CLASSES = 10


def load_data(
    batch_size: int = BATCH_SIZE,
    buffer_size: int = BUFFER_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load the mnist dataset

    Returns:
        ((tf.Tensor, tf.Tensor), (tf.Tensor, tf.Tensor)): The training
            and test data as a tuple.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.shuffle(buffer_size).batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds


class Model(tf.keras.Model):
    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_layers: int = 3,
        units: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv_layers = [
            tf.keras.layers.Conv2D(32, 3, activation='relu')
            for _ in range(n_layers)
        ]
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.out = tf.keras.layers.Dense(n_classes)

    def call(
        self,
        x: tf.TensorArray,
        training: bool = False
    ) -> tf.TensorArray:
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out(x)
        return x


def main() -> int:
    train_ds, test_ds = load_data()

    model = Model(n_classes=N_CLASSES)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Train metrics.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

    # Test metrics.
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    @tf.function
    def train_step(x: tf.TensorArray, y: tf.TensorArray) -> None:
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = loss_fn(y, pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_acc(y, pred)

    @tf.function
    def test_step(images: tf.TensorArray, labels: tf.TensorArray) -> None:
        pred = model(images, training=False)
        loss = loss_fn(labels, pred)

        test_loss(loss)
        test_acc(labels, pred)

    for epoch in range(EPOCHS):
        # Reset metrics at the start of each epoch.
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_state()
        test_acc.reset_state()

        # Train loop.
        for images, labels in train_ds:
            train_step(images, labels)

        # Test loop.
        for images, labels in test_ds:
            test_step(images, labels)

        print(f'''\
Epoch:      {epoch + 1}
Train loss: {train_loss.result():.4f}
Train acc:  {train_acc.result():.2%}
Test loss:  {test_loss.result():.4f}
Test acc:   {test_acc.result():.2%}
''')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
