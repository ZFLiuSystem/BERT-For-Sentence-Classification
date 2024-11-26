from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from config import Args


class Processor:
    @staticmethod
    def read_txt(filepath):
        with open(filepath, encoding='utf-8') as f:
            raw_examples = f.read().strip().split('\n')
        return raw_examples

    @staticmethod
    def get_examples(raw_examples):
        contents = []
        labels = []
        for line in raw_examples:
            line = line.split('\t')
            contents.append(line[0])
            labels.append(line[1])
        return contents, labels


class TextClassificationDataset(Dataset):
    def __init__(self, samples, labels):
        self.input_ids = samples['input_ids']
        self.attention_mask = samples['attention_mask']
        self.labels = labels
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ids):
        length = self.input_ids.size(0)
        assert length >= ids + 1
        input_samples = {
            'input_ids': self.input_ids[ids],
            'attention_mask': self.attention_mask[ids]
        }
        return input_samples, self.labels[ids]


def get_set(file_path, model_path):
    args_instance = Args()
    args = args_instance.get_parser()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    processor = Processor()
    raw_examples = processor.read_txt(file_path)
    contents, labels_list = processor.get_examples(raw_examples)
    samples = tokenizer.batch_encode_plus(contents,
                                          add_special_tokens=True,
                                          max_length=args.max_length,
                                          padding='max_length',
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')
    labels = []
    for label in labels_list:
        label = float(label)
        labels.append(label)
    labels = torch.tensor(labels, dtype=torch.int64)
    return samples, labels
    pass
