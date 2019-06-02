import datetime

import tensorflow as tf

from classifier.bert import run_classifier_with_tfhub
from classifier.constants import TRAIN_BATCH_SIZE, SAVE_CHECKPOINTS_STEPS, BERT_INIT_CHECKPOINT, MAX_SEQ_LENGTH, \
    BERT_MODEL_HUB, NUM_TRAIN_EPOCHS, CHECKPOINT_DIR, BASE_DIR, EXPORT_DIR
from classifier.dataset import ToxicCommentsProcessor, convert_examples_to_features, input_fn_builder, \
    serving_input_receiver_fn
from classifier.facade import BERTFacade


def get_toxic_comments_estimator(
        num_train_examples,
        output_dir,
        init_checkpoint=BERT_INIT_CHECKPOINT
):
    labels = ToxicCommentsProcessor().get_labels()
    num_train_steps = int(num_train_examples / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)

    model_fn = BERTFacade().build_model_fn(
        len(labels), num_train_steps, init_checkpoint
    )

    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            model_dir=output_dir,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        ),
        params={"batch_size": TRAIN_BATCH_SIZE},
    )


def model_train(estimator, train_examples):
    print('Creating tokenizer from tf hub')

    tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)

    print('Converting examples to features...')

    train_features = convert_examples_to_features(train_examples, MAX_SEQ_LENGTH, tokenizer)

    num_train_examples = len(train_examples)
    num_train_steps = int(num_train_examples / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)

    print('***** Started training at {} *****'.format(datetime.datetime.now()))
    print(f' Num train examples = {num_train_examples}')
    print(f' Batch size = {TRAIN_BATCH_SIZE}')
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True
    )

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print('***** Finished training at {} *****'.format(datetime.datetime.now()))


def model_export():
    processor = ToxicCommentsProcessor()
    train_examples = processor.get_train_examples(BASE_DIR)
    estimator = get_toxic_comments_estimator(len(train_examples), CHECKPOINT_DIR)
    estimator.export_saved_model(EXPORT_DIR, serving_input_receiver_fn=serving_input_receiver_fn)
