import tensorflow as tf

from classifier.constants import ACCURACY_THRESHOLD
from .bert import modeling


def create_model(
        bert_config,
        is_training,
        input_ids,
        input_mask,
        segment_ids,
        labels,
        num_labels,
        use_one_hot_embeddings
):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.nn.sigmoid(logits)  # multi-label case

        labels = tf.cast(labels, tf.float32)
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)
        tf.summary.scalar('loss', loss)

        logits_split = tf.split(probabilities, num_labels, axis=-1)
        label_ids_split = tf.split(labels, num_labels, axis=-1)
        for j, logits in enumerate(logits_split):
            label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
            current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
            accuracy = tf.reduce_mean(
                tf.where(logits > ACCURACY_THRESHOLD,
                tf.ones_like(logits, dtype=tf.int32),
                tf.zeros_like(logits, dtype=tf.int32)))
            tf.summary.scalar('auc_{}'.format(j), current_auc)
            tf.summary.scalar('accuracy_{}'.format(j), accuracy)

        return loss, per_example_loss, logits, probabilities
