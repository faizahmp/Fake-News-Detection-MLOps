# %% [markdown]
# # Library

# %%
# !pip install tfx

# %%
from tfx.proto import pusher_pb2
from tfx.components import Pusher
from tfx.components import Evaluator
import tensorflow_model_analysis as tfma
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.types import Channel
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.dsl.components.common.resolver import Resolver
from tfx.proto import trainer_pb2
from fakenews_transform import (
    LABEL_KEY,
    transformed_name
)
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow_transform as tft
import pandas as pd
import zipfile
import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import os

# %% [markdown]
# # PIPELINES

# %% [markdown]
# ## Dataset Kaggle

# %%
# !pip install kaggle

# %%
# !mkdir ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
import os
import shutil

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
shutil.copy("kaggle.json", kaggle_dir)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# %%
# ! kaggle datasets download saurabhshahane/fake-news-classification

# %%
with zipfile.ZipFile("fake-news-classification.zip", 'r') as zip_ref:
    zip_ref.extractall("E:\\submission1\\dataset")

print(f"Files extracted to E:\\submission1\\dataset")

# %%
dataset = pd.read_csv("./dataset/WELFake_Dataset.csv")
dataset.info()

# %%
dataset = dataset.drop(columns=["Unnamed: 0"])

# %%
dataset.info()

# %%
dataset.isnull().sum()

# %%
dataset = dataset.dropna()

# %%
dataset = dataset.to_csv('./dataset/WELFake_Dataset.csv', index=False)

# %% [markdown]
# ## Set Variable

# %%
DATA_ROOT = './dataset'

# %%
PIPELINE_NAME = "faizahmp-pipeline"
SCHEMA_PIPELINE_NAME = "fakenews-schema"

# Dir untuk menyimpan artifact yang akan dihasilkan
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)
# from absl import logging
# logging.set_verbosity(logging.INFO)

# %% [markdown]
# ## Data Ingestion

# %%
# ExampleGen
output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(
            name="train", hash_buckets=8),  # split train eval 80:20
        example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
    ])
)
example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)

# %%
interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)

# %%
interactive_context.run(example_gen)

# %% [markdown]
# ## Data Validation

# %%
# StatisticsGen: menampilkan summary statistic
statistics_gen = StatisticsGen(
    examples=example_gen.outputs["examples"]
)
interactive_context.run(statistics_gen)

# %%
interactive_context.show(statistics_gen.outputs["statistics"])

# %%
# data schema
schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
interactive_context.run(schema_gen)

# %%
interactive_context.show(schema_gen.outputs["schema"])

# %%
# ExampleValidator --> buat detect anomali
# inputan needs dari statistics & schema
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
interactive_context.run(example_validator)

# %%
interactive_context.show(example_validator.outputs["anomalies"])

# %% [markdown]
# # Data Pre-Processing

# %%
# nama dari module
TRANSFORM_MODULE_FILE = "fakenews_transform.py"

# %%
# % % writefile {TRANSFORM_MODULE_FILE}
LABEL_KEY = "label"
FEATURE_KEY = "title"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        inputs[LABEL_KEY], tf.float32)

    return outputs


# %%
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)
interactive_context.run(transform)

# %% [markdown]
# # Model Training

# %% [markdown]
# ## Training

# %%
TRAINER_MODULE_FILE = "fakenews_trainer.py"

# %%
# % % writefile {TRAINER_MODULE_FILE}

LABEL_KEY = "label"
FEATURE_KEY = "title"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern,
             tf_transform_output,
             num_epochs,
             batch_size=64) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

# os.environ['TFHUB_CACHE_DIR'] = '/hub_chace'
# embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")


# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)


embedding_dim = 16


def model_builder():
    """Build machine learning model"""
    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy()]

    )

    # print(model)
    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):

        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)]])

    # Build the model
    model = model_builder()

    # Train the model
    model.fit(x=train_set,
              validation_data=val_set,
              callbacks=[tensorboard_callback, es, mc],
              steps_per_epoch=1000,
              validation_steps=1000,
              epochs=10)
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)


# %%

trainer = Trainer(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)
interactive_context.run(trainer)

# %% [markdown]
# # Analysis and Validation

# %% [markdown]
# ## Resolver

# %%

model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')

interactive_context.run(model_resolver)

# %% [markdown]
# ## Evaluator

# %%
# Evaluator

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[

            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={'value': 0.5}),
                                  change_threshold=tfma.GenericChangeThreshold(
                                      direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute={'value': 0.0001})
                              )
                              )
        ])
    ]

)

# %%
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)

interactive_context.run(evaluator)

# %%
# Visualisasi hasil evaluation
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)

# %% [markdown]
# ## Pusher

# %%

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model_dir/fakenews-detection-model'))
)

interactive_context.run(pusher)
