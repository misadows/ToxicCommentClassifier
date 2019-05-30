import os

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
MAX_SEQ_LENGTH = 128

WARMUP_PROPORTION = 0.1

SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 1000

BUCKET = 'smucha_colab'
OUTPUT_DIR = f'gs://{BUCKET}/bert-tfhub/models/'

BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_MODEL_HUB = f'https://tfhub.dev/google/bert_{BERT_MODEL}/1'

BERT_PRETRAINED_DIR = f'gs://cloud-tpu-checkpoints/bert/{BERT_MODEL}'
BERT_INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
