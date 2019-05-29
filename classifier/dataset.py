import pandas as pd


class InputExample(object):
    """A single training/test example for sequence multi-label classification."""

    def __init__(self, guid, text, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the sequence.
            labels: The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class ToxicCommentsProcessor:
    TRAIN_VAL_RATIO = 0.9

    def get_train_examples(self, data_dir):
        dataframe = pd.read_csv(f'{data_dir}/train.csv')
        train_size = self._get_train_size(len(dataframe))
        train_df = dataframe[:train_size]
        return self._create_examples(train_df)

    def get_dev_examples(self, data_dir):
        dataframe = pd.read_csv(f'{data_dir}/train.csv')
        train_size = self._get_train_size(len(dataframe))
        dev_df = dataframe[train_size:]
        return self._create_examples(dev_df)

    def get_test_examples(self, data_dir):
        dataframe = pd.read_csv(f'{data_dir}/test.csv')
        return self._create_examples(dataframe)

    def get_labels(self):
        return ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def _get_train_size(self, total_length):
        return int(self.TRAIN_VAL_RATIO * total_length)

    def _create_examples(self, df, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text = row[1]
            if labels_available:
                labels = row[2:]
            else:
                labels = [0, 0, 0, 0, 0, 0]
            examples.append(InputExample(guid=guid, text=text, labels=labels))
        return examples
