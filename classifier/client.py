import requests
import json

from classifier.bert import run_classifier_with_tfhub
from classifier.constants import BASE_DIR, BERT_MODEL_HUB, MAX_SEQ_LENGTH
from classifier.dataset import ToxicCommentsProcessor, convert_examples_to_features

processor = ToxicCommentsProcessor()
dev_examples = processor.get_dev_examples(BASE_DIR)

tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)
dev_features = convert_examples_to_features(dev_examples, MAX_SEQ_LENGTH, tokenizer)

example = dev_features[0]
data = {
    'instances': [
        {
            'input_ids': example.input_ids,
            'input_mask': example.input_mask,
            'segment_ids': example.segment_ids,
        }
    ]
}

url = 'http://localhost:8501/v1/models/saved_model_basic:predict'
r = requests.post(url, data=json.dumps(data))
print(r.text)
