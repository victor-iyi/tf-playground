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

import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


DATA_URL = ('https://storage.googleapis.com/download.tensorflow.org/'
            'example_images/flower_photos.tgz')
SUNFLOWER_URL = ('https://storage.googleapis.com/download.tensorflow.org/'
                 'example_images/592px-Red_sunflower.jpg')
TF_LITE_MODEL_PATH = '../../saved_models/flower-classifier.tflite'
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL = 512, 512, 3
EPOCHS, BATCH_SIZE, LEARNING_RATE = 15, 32, 1e-3
N_CLASSES, FIG_SIZE = len(CLASS_NAMES), (5, 5)


def load_data(
    data_dir: str,
    url: str = DATA_URL,
    train: bool = True,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tf.data.Dataset:
    """Download and load data into a TF dataset

    Args:
        data_dir (str): Dataset directory.
        url (str, optional): URL to download data from.
            Defaults to DATA_URL.
        train (bool, optional): Load training if set to True otherwise,
            loads validation data. Defaults to True.
        image_size (tuple[int, int], optional): Image size in form of
            height by width. Defaults to (224, 224).
        batch_size (int, optional): Mini batch size. Defaults to 32.
        validation_split (float, optional): Validation split ratio.
            Defaults to 0.2.
        seed (int, optional): Seed random number. Defaults to 42.

    Returns:
        tf.data.Dataset: Loaded dataset.
    """
    data_dir = tf.keras.utils.get_file(data_dir, origin=url, untar=True)
    data_path = pathlib.Path(data_dir)

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset='training' if train else 'validation',
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    return dataset


def visualize_data(
    dataset: tf.data.Dataset,
    class_names: list[str] = CLASS_NAMES,
    figsize: tuple[int, int] = FIG_SIZE,
) -> None:
    """Visualize dataset.

    Args:
        dataset (tf.data.Dataset): Dataset to visualize.
        class_names (list[str]): Class names.
        figsize (tuple[int, int], optional): Figure size. Defaults to (5, 5).
    """

    plt.figure(figsize=figsize)

    for img, label in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(class_names[label[i]])
            plt.axis('off')
    plt.show()


def visualize_train_history(
    history: tf.keras.callbacks.History,
    epochs: int = EPOCHS,
    figsize: tuple[int, int] = FIG_SIZE,
) -> None:
    """Visualize the training & validation loss and accuracy.

    Args:
        history (tf.keras.callbacks.History): Model training history object.
        epochs (int): Epochs used for training. Defaults to EPOCHS.
        figsize (tuple[int, int], optional): Figure size.
            Defaults to (5, 5).
    """
    # Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epoch_range = range(epochs)
    plt.figure(figsize=figsize)

    # Training & validation accuracy.
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, acc, label='Training Accuracy')
    plt.plot(epoch_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training & Validation Accuracy')

    # Training & validation loss.
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, loss, label='Training Loss')
    plt.plot(epoch_range, val_loss, label='Validation Loss')
    plt.legend(loc='lower left')
    plt.title('Training & Validation Loss')

    plt.show()


class DataAugmentation(tf.keras.layers.Layer):

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH,
                                             IMG_CHANNEL),
    ) -> None:
        """Data augmentation layer.

        Args:
            input_shape (tuple, optional): Input shape.
                Defaults to (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL).
        """
        super(DataAugmentation, self).__init__()

        self.data_augmentation = [
            tf.keras.layers.RandomFlip(
                mode='horizontal', input_shape=input_shape,
            ),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.Rescaling(1. / 255),
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Call method.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Training flag. Defaults to False.

        Returns:
            tf.Tensor: Augmented tensor.
        """
        if training:
            for layer in self.data_augmentation:
                inputs = layer(inputs)
        return inputs


class ImageClassification(tf.keras.Model):
    def __init__(
        self,
        n_classes: int = N_CLASSES,
        input_shape: tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH,
                                             IMG_CHANNEL),
        dropout: float = 0.2,
    ) -> None:
        super(ImageClassification, self).__init__()

        self.augmentation = DataAugmentation(input_shape=input_shape)
        self.conv_layers = [
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
        ]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(n_classes, name='output')

    def call(
        self,
        inputs: tf.TensorArray,
        training: bool = False
    ) -> tf.TensorArray:
        x = self.augmentation(inputs, training=training)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.output_layer(x)

        return x


def inference(
    model: tf.keras.Model,
    img_path: str,
    class_names: list[str] = CLASS_NAMES,
    target_size: tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
) -> None:
    """Perform inference on image.

    Args:
        model (tf.keras.Model): Trained model.
        img_path (str): Image path to predict.
        class_names (list[str]): List of class names.
        target_size (tuple, optional): Target image size
            (must be same as used in the trained model).
            Defaults to (IMG_HEIGHT, IMG_WIDTH).
    """
    img = tf.keras.utils.load_img(
        img_path, target_size=target_size
    )
    img_arr = tf.keras.utils.img_to_array(img)
    img_arr = tf.expand_dims(img_arr, 0)  # Create a batch.

    predictions = model.predict(img_arr)
    score = tf.nn.softmax(predictions[0])

    print(f'This image is most likely to be {class_names[np.argmax(score)]}'
          f' with a {np.max(score):.02%} confidence.')


def convert_to_tflite(
    model: tf.keras.Model,
    tflite_path: str,
) -> None:
    """Convert Keras model to TF Lite for on-device inference.

    Args:
        model (tf.keras.Model): Keras model.
        tflite_path (str): Path to save the TF Lite model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Create TF lite saved directory if it doesn't exist.
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

    # Save the TFLite model.
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)


def inference_with_tflite(
    img_url: str,
    model_path: str = TF_LITE_MODEL_PATH,
    class_names: list[str] = CLASS_NAMES,
    signature: str = 'serving_default',
    target_size: tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
) -> None:
    """Perform inference on image with TF Lite model.

    Args:
        img_url (str): Path to image to predict.
        model_path (str): Path to saved TF Lite model.
            Defaults to TF_LITE_MODEL_PATH.
        class_names (list[str]): List of class names.
            Defaults to CLASS_NAMES.
        signature (str, optional): TF Lite model signature.
            Defaults to 'serving_default'.
        target_size (tuple, optional): Image target size.
            Defaults to (IMG_HEIGHT, IMG_WIDTH).
    """
    # Load the model with `Interpreter`
    interpreter = tf.lite.Interpreter(model_path=model_path)
    classify_lite = interpreter.get_signature_runner(signature)
    print(interpreter.get_signature_list())

    img_path = tf.keras.utils.get_file('Red_sunflower', origin=img_url)
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_arr = tf.keras.utils.img_to_array(img)
    img_arr = tf.expand_dims(img_arr, 0)  # create a batch.

    pred_lite = classify_lite(sequential_1_input=img_arr)['output']
    score_lite = tf.nn.softmax(pred_lite)

    print('This image most likely belongs to '
          f'{class_names[np.argmax(score_lite)]}'
          f' with a {np.max(score_lite):.02%} confidence.')


def main() -> int:
    print(f'Using TensorFlow version: {tf.__version__}')

    # Load training & validation data.
    train_ds = load_data('flower_photos', train=True)
    val_ds = load_data('flower_photos', train=False)

    # Get the class names.
    # class_names = train_ds.class_names
    # print(class_names)

    # Pre-process data.
    train_ds = (train_ds.cache()
                .shuffle(1000)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
    val_ds = (val_ds
              .cache()
              .prefetch(buffer_size=tf.data.AUTOTUNE))

    # Visualize some training samples.
    visualize_data(train_ds, CLASS_NAMES)

    # Create the model.
    model = ImageClassification(n_classes=N_CLASSES)
    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train the model.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Visualize training history.
    visualize_train_history(history, epochs=EPOCHS)

    # Perform inference.
    sunflower_path = tf.keras.utils.get_file(
        'Red_sunflower', origin=SUNFLOWER_URL
    )
    inference(model, sunflower_path, CLASS_NAMES)

    # Convert to TF Lite.
    convert_to_tflite(model, TF_LITE_MODEL_PATH)

    # Perform inference with TF Lite.
    inference_with_tflite(img_url=SUNFLOWER_URL,
                          model_path=TF_LITE_MODEL_PATH,
                          class_names=CLASS_NAMES)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
