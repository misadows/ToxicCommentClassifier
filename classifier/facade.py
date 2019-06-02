import os
from dataclasses import dataclass

import tensorflow as tf

from .bert import modeling, optimization
from .constants import (
    BERT_PRETRAINED_DIR, LEARNING_RATE, WARMUP_PROPORTION,
    BERT_INIT_CHECKPOINT)
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

    def build_model_fn(self, num_labels, num_train_steps, init_checkpoint=BERT_INIT_CHECKPOINT):
        bert_config = self.get_config()

        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

        model_config = ModelConfig(
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            num_labels=num_labels,
            use_tpu=False,
            use_one_hot_embeddings=False
        )

        return self._create_model_fn(bert_config, model_config, init_checkpoint)

    def _create_model_fn(
            self,
            bert_config,
            model_config: ModelConfig,
            init_checkpoint=None,
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

            tvars = tf.trainable_variables()
            assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            estimator_spec_factory = EstimatorSpecAbstractFactory(model_config).create(mode)

            return estimator_spec_factory.create(model_nodes, features)

        return model_fn


class EstimatorTrainSpecFactory:

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    def create(self, model_nodes, features):
        total_loss, _, _, _ = model_nodes

        train_op = optimization.create_optimizer(
            total_loss,
            self._model_config.learning_rate,
            self._model_config.num_train_steps,
            self._model_config.num_warmup_steps,
            self._model_config.use_tpu,
        )

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=total_loss,
            train_op=train_op,
        )


class EstimatorEvalSpecFactory:

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    def create(self, model_nodes, features):
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

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=total_loss,
            eval_metrics=eval_metrics,
        )


class EstimatorDefaultSpecFactory:

    def __init__(self, mode):
        self.mode = mode

    def create(self, model_nodes, features):
        _, _, _, probabilities = model_nodes

        return tf.estimator.EstimatorSpec(
            mode=self.mode,
            predictions={"probabilities": probabilities},
        )


class EstimatorSpecAbstractFactory:

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    def create(self, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return EstimatorTrainSpecFactory(self._model_config)

        if mode == tf.estimator.ModeKeys.EVAL:
            return EstimatorEvalSpecFactory(self._model_config)

        return EstimatorDefaultSpecFactory(mode)
