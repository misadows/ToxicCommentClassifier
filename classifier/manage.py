import tensorflow as tf

from classifier.constants import TRAIN_BATCH_SIZE, SAVE_CHECKPOINTS_STEPS, BERT_INIT_CHECKPOINT
from classifier.dataset import ToxicCommentsProcessor
from classifier.facade import BERTFacade


def get_toxic_comments_estimator(
        task_data_dir,
        output_dir,
        init_checkpoint=BERT_INIT_CHECKPOINT
):
    processor = ToxicCommentsProcessor()

    train_examples = processor.get_train_examples(task_data_dir)
    labels = processor.get_labels()

    model_fn = BERTFacade().build_model_fn(
        len(train_examples), len(labels), init_checkpoint
    )

    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            model_dir=output_dir,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        ),
        params={"batch_size": TRAIN_BATCH_SIZE},
    )
