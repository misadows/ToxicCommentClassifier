import os
from dataclasses import dataclass

import tensorflow as tf

from .bert import modeling, optimization
from .constants import (
    BERT_PRETRAINED_DIR, OUTPUT_DIR, LEARNING_RATE, TRAIN_BATCH_SIZE, NUM_TRAIN_EPOCHS,
    WARMUP_PROPORTION
)
from .model import create_model


@dataclass
class ModelConfig:
    learning_rate: float
    num_train_steps: int
    num_warmup_steps: int
    num_labels: int
    use_tpu: bool
    use_one_hot_embeddings: bool


class BERTFacade:

    def get_config(self):
        config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
        return modeling.BertConfig.from_json_file(config_file)

    def build_model_fn(self, num_train_examples, num_labels):
        bert_config = self.get_config()
        init_checkpoint = tf.train.latest_checkpoint(OUTPUT_DIR, latest_filename=None)

        num_train_steps = int(num_train_examples / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

        model_config = ModelConfig(
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            num_labels=num_labels,
            use_tpu=True,
            use_one_hot_embeddings=False
        )

        return self._create_model_fn(bert_config, model_config, init_checkpoint)

    def _create_model_fn(
            self,
            bert_config,
            model_config: ModelConfig,
            init_checkpoint,
    ):
        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            input_ids, input_mask = features["input_ids"], features["input_mask"]
            segment_ids, label_ids = features["segment_ids"], features["label_ids"]
            num_labels = model_config.num_labels
            use_one_hot_embeddings = model_config.use_one_hot_embeddings

            is_training = mode == tf.estimator.ModeKeys.TRAIN

            model_nodes = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings
            )

            scaffold_fn = TPUScaffoldFactory().create_scaffold_fn(init_checkpoint)
            estimator_spec_factory = TPUEstimatorSpecAbstractFactory(model_config).create(mode)

            return estimator_spec_factory.create(model_nodes, features, scaffold_fn)

        return model_fn


class TPUEstimatorTrainSpecFactory:

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    def create(self, model_nodes, features, scaffold_fn):
        total_loss, _, _, _ = model_nodes

        train_op = optimization.create_optimizer(
            total_loss,
            self._model_config.learning_rate,
            self._model_config.num_train_steps,
            self._model_config.num_warmup_steps,
            self._model_config.use_tpu,
        )

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn
        )


class TPUEstimatorEvalSpecFactory:

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    def create(self, model_nodes, features, scaffold_fn):
        total_loss, per_example_loss, _, probabilities = model_nodes

        def metric_fn(per_example_loss, label_ids, probabilities):
            num_labels = self._model_config.num_labels
            logits_split = tf.split(probabilities, num_labels, axis=-1)
            label_ids_split = tf.split(label_ids, num_labels, axis=-1)
            # metrics change to auc of every class
            eval_dict = {}
            for j, logits in enumerate(logits_split):
                label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                eval_dict[str(j)] = (current_auc, update_op_auc)
            eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
            return eval_dict

        label_ids = features['label_ids']
        eval_metrics = (metric_fn, [per_example_loss, label_ids, probabilities])

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn
        )


class TPUEstimatorDefaultSpecFactory:

    def __init__(self, mode):
        self.mode = mode

    def create(self, model_nodes, features, scaffold_fn):
        _, _, _, probabilities = model_nodes

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=self.mode,
            predictions={"probabilities": probabilities},
            scaffold_fn=scaffold_fn
        )


class TPUEstimatorSpecAbstractFactory:

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    def create(self, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return TPUEstimatorTrainSpecFactory(self._model_config)

        if mode == tf.estimator.ModeKeys.EVAL:
            return TPUEstimatorEvalSpecFactory(self._model_config)

        return TPUEstimatorDefaultSpecFactory(mode)


class TPUScaffoldFactory:

    def create_scaffold_fn(self, init_checkpoint=None):
        if not init_checkpoint:
            return None

        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

        return tpu_scaffold
