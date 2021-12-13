from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, InputLayer
from tensorflow.keras.models import Sequential
import os
import argparse
import mlflow
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tensorflow_datasets.image_classification import Cifar10 as DatasetClass
import tensorflow_datasets as tfds
import numpy as np
import io
import cv2

ds_info = DatasetClass().info

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

parser = argparse.ArgumentParser(prog=Path(
    __file__).name, description="{}のスクラッチ学習".format(ds_info.name), add_help=True)

parser.add_argument("-s", "--sample_usage_rate", type=int, choices=range(1, 81),
                    metavar="[1-80]", required=True, help="データセット全体から訓練に使うサンプルの割合")

args = parser.parse_args()
SAMPLE_USAGE_RATE = args.sample_usage_rate


def preprocess_dataset(image: tf.Tensor, label: tf.Tensor):
    image = tf.cast(image, dtype=tf.float32)/255. - 0.50
    return (image, label)


mlflow.log_param("Sample Usage Rate", SAMPLE_USAGE_RATE * 0.01)

train_ds, test_ds = tfds.load(ds_info.name, split=["train[:{}%]".format(
    SAMPLE_USAGE_RATE), "train[80%:]"], as_supervised=True)

INPUT_SHAPE = ds_info.features["image"].shape
NUM_CLASSES = ds_info.features["label"].num_classes
NUM_EXAMPLES = ds_info.splits["train"].num_examples
BATCH_SIZE = 256

print("Number of Class: {}, Input Shape: {}, Batch Size: {}".format(
    NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE))

def save_to_mlflow_data_examples(ds: tf.data.Dataset):
    # ローカル実行では、画像ビューワが立ち上がる。show_examples内のply.showをコメントアウトすべし。
    fig = tfds.show_examples(train_ds, ds_info, rows=4, cols=4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, )
    buf.seek(0)
    examples = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    examples = cv2.imdecode(examples, cv2.IMREAD_UNCHANGED)

    mlflow.log_image(examples, "examples.png")

save_to_mlflow_data_examples(train_ds)

train_ds = train_ds.map(preprocess_dataset)
test_ds = test_ds.map(preprocess_dataset)

train_ds = train_ds.shuffle(NUM_EXAMPLES//4).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

model = Sequential(
    [
        InputLayer(input_shape=INPUT_SHAPE),
        Conv2D(16, kernel_size=3),
        MaxPool2D(),
        Conv2D(32, kernel_size=3),
        Flatten(),
        Dense(128),
        Dense(64),
        Dense(units=NUM_CLASSES)
    ]
)
model.summary()

model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(
    from_logits=True), metrics=["accuracy"])

mlflow.tensorflow.autolog()
model.fit(x=train_ds, validation_data=test_ds,
          epochs=10, batch_size=BATCH_SIZE)
