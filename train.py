import os
import argparse
import mlflow
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tensorflow_datasets.image_classification import Cifar10 as DatasetClass
ds_info = DatasetClass().info

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

with mlflow.start_run():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("{}-classification".format(ds_info.name))

    parser = argparse.ArgumentParser(prog=Path(__file__).name, description="{}のスクラッチ学習".format(ds_info.name), add_help=True)

    parser.add_argument("-s", "--sample_usage_rate", type = int, choices=range(1, 81), metavar="[1-80]", required=True, help="データセット全体から訓練に使うサンプルの割合")

    args = parser.parse_args()
    SAMPLE_USAGE_RATE = args.sample_usage_rate
    TARGET_SIZE = 224
    def preprocess_dataset(image:tf.Tensor, label:tf.Tensor):
        image = tf.cast(image, dtype=tf.float32)/255. - 0.50
        return (image, label)

    mlflow.log_param("sample_usage_rate", SAMPLE_USAGE_RATE * 0.01)

    train_ds, test_ds = tfds.load(ds_info.name, split=["train[:{}%]".format(SAMPLE_USAGE_RATE), "train[80%:]"], as_supervised=True)

    INPUT_SHAPE = ds_info.features["image"].shape
    NUM_CLASSES = ds_info.features["label"].num_classes
    NUM_EXAMPLES = ds_info.splits["train"].num_examples
    BATCH_SIZE = 256

    print("Number of Class: {}, Input Shape: {}, Batch Size: {}".format(NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE))

    train_ds = train_ds.map(preprocess_dataset)
    test_ds = test_ds.map(preprocess_dataset)

    train_ds = train_ds.shuffle(NUM_EXAMPLES//4).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)

    import tensorflow_hub as hub
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape = INPUT_SHAPE)
    feature_extractor.trainable = False

    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE),
            Conv2D(16, kernel_size=3),
            MaxPool2D(),
            Conv2D(32, kernel_size=3),
            Flatten(),
            Dense(128),
            Dense(64),
            Dense(units = NUM_CLASSES)
        ]
    )
    model.summary()

    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    mlflow.tensorflow.autolog()
    history = model.fit(x = train_ds, validation_data=test_ds, epochs=3, batch_size=BATCH_SIZE)

