import tensorflow as tf

from classifier.constants import TRAIN_BATCH_SIZE, OUTPUT_DIR, SAVE_CHECKPOINTS_STEPS
from classifier.dataset import ToxicCommentsProcessor
from classifier.facade import BERTFacade


def get_toxic_comments_estimator(task_data_dir):
    processor = ToxicCommentsProcessor()

    train_examples = processor.get_train_examples(task_data_dir)
    labels = processor.get_labels()

    model_fn = BERTFacade().build_model_fn(len(train_examples), len(labels))

    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            model_dir=OUTPUT_DIR,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        ),
        params={"batch_size": TRAIN_BATCH_SIZE},
    )
