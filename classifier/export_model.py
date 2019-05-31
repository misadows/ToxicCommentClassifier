import tensorflow as tf

from classifier.bert import run_classifier_with_tfhub
from classifier.constants import MAX_SEQ_LENGTH, BERT_MODEL_HUB
from classifier.manage import get_toxic_comments_estimator
from classifier.dataset import ToxicCommentsProcessor, convert_text_to_bert_features


def serving_input_receiver_fn():
    feature_spec = {
        'text': tf.FixedLenFeature([], dtype=tf.string)
    }

    default_batch_size = 1

    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[default_batch_size],
        name='input_text_tensor')

    received_tensors = {'text': serialized_tf_example}

    tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)
    preprocess_fn = lambda text: convert_text_to_bert_features(text, MAX_SEQ_LENGTH, tokenizer)
    input_ids, input_mask, segment_ids = tf.map_fn(preprocess_fn, serialized_tf_example)

    features = {
        "input_ids":
            tf.constant(
                    input_ids, shape=[default_batch_size, MAX_SEQ_LENGTH],
                    dtype=tf.int32),
        "input_mask":
            tf.constant(
                input_mask,
                shape=[default_batch_size, MAX_SEQ_LENGTH],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                segment_ids,
                shape=[default_batch_size, MAX_SEQ_LENGTH],
                dtype=tf.int32),
        }

    return tf.estimator.export.ServingInputReceiver(features, received_tensors)


processor = ToxicCommentsProcessor()

BASE_DIR = ''
OUTPUT_DIR = ''
EXPORT_DIR = ''

train_examples = processor.get_train_examples(BASE_DIR)
estimator = get_toxic_comments_estimator(len(train_examples), OUTPUT_DIR)

estimator.export_saved_model(EXPORT_DIR, serving_input_receiver_fn=serving_input_receiver_fn)
