import sys
import requests


PREDICT_URL = "http://34.65.73.45:8010/predict"
BATCH_SIZE = 8


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return [line.rstrip('\n') for line in file]


def predict_sentences(sentences):
    data = {
        "instances": [{"text": sentence} for sentence in sentences]
    }
    response = requests.post(PREDICT_URL, json=data)
    response_data = response.json()
    return response_data["predictions"]


def show_labels(predictions):
    for prediction in predictions:
        labels = ['1' if prob > 0.5 else '0' for prob in prediction]
        line = ' '.join(labels)
        print(line)


def main():
    try:
        dataset_path = sys.argv[1]
    except IndexError:
        print('dataset_path argument missing - `python client.py <dataset_path>`')
        return

    sentences = load_dataset(dataset_path)

    for sentence_batch in batch(sentences, BATCH_SIZE):
        predictions = predict_sentences(sentence_batch)
        show_labels(predictions)


if __name__ == '__main__':
    main()
