import json

import requests

from flask import Flask, request, Response

from classifier.bert import run_classifier_with_tfhub
from classifier.constants import BERT_MODEL_HUB, MAX_SEQ_LENGTH
from classifier.dataset import convert_examples_to_features, InputExample

MODEL_URL = 'http://34.65.73.45:8501/v1/models/saved_model_basic:predict'

app = Flask(__name__)


def preprocess_text_instances(instances):
    examples = [InputExample(i, inst["text"], [0, 0, 0, 0, 0, 0]) for i, inst in enumerate(instances)]

    tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)
    features = convert_examples_to_features(examples, MAX_SEQ_LENGTH, tokenizer)

    return [
        {
            'input_ids': example.input_ids,
            'input_mask': example.input_mask,
            'segment_ids': example.segment_ids,
        } for example in features
    ]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    preprocessed_instances = preprocess_text_instances(data["instances"])

    response = requests.post(MODEL_URL, data=json.dumps({"instances": preprocessed_instances}))

    return Response(response.content, response.status_code)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010)
