import tensorflow as tf


def load_data() -> tuple[
    tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]
]:
    """Load the mnist dataset

    Returns:
        ((tf.Tensor, tf.Tensor), (tf.Tensor, tf.Tensor)): The training
            and test data as a tuple.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_model(
    input_shape: tuple[int, int] = (28, 28),
    n_classes: int = 10,
    units: int = 128,
    dropout: float = 0.2,
    activation: str = 'relu',
) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(units, activation=activation),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_classes)
    ])

    return model


def main() -> int:
    print(f'Using TensorFlow version: {tf.__version__}')
    (x_train, y_train), (x_test, y_test) = load_data()

    print(f'x_train shape: {x_train.shape} | y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape} | y_test shape: {y_test.shape}')

    model = build_model()
    predictions = model(x_train[:1])
    print(f'Prediction: {predictions.numpy()}')

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(y_train[:1], predictions)
    print(f'Loss: {loss.numpy()}')

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, epochs=5)

    result = model.evaluate(x_test, y_test)
    print(f'Evaluation results: {result}')

    probability_model = tf.keras.models.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    predictions = probability_model(x_test[:5])
    print(f'Predictions: {predictions.numpy()}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
