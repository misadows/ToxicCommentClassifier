import tensorflow as tf

from .constants import SAVE_CHECKPOINTS_STEPS, ITERATIONS_PER_LOOP, NUM_TPU_CORES, OUTPUT_DIR


class TPURunConfig:

    def __init__(self, tpu_cluster_resolver):
        self._tpu_cluster_resolver = tpu_cluster_resolver

    def get(self):
        return tf.contrib.tpu.RunConfig(
            cluster=self._tpu_cluster_resolver,
            model_dir=OUTPUT_DIR,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=ITERATIONS_PER_LOOP,
                num_shards=NUM_TPU_CORES,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
            )
        )

