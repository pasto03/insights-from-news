import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import config
import pickle


def split_dataset_to_chunks(dataset, chunk_spacing=1, days_before=30, day_prediction=7, drop_last=True):
    chunk_size = days_before + day_prediction
    raw_chunks = [dataset[i: i+chunk_size] for i in range(0, len(dataset) - (chunk_size + 1))]
    # (features, targets) in list
    chunks = [(chunk[:days_before], chunk[days_before:chunk_size]) for chunk in raw_chunks][::chunk_spacing]
    return chunks[:-1] if drop_last else chunks


merged_json = config.merged_json


class NewsAndStocks(Dataset):
    def __init__(self, train=True, chunk_spacing=1):
        train_size = 0.7
        self.dataset = merged_json[::-1]  # 顺序排列
        self.__split_idx = int(train_size * len(self.dataset))
        self.data = self.dataset[:self.__split_idx] if train else self.dataset[self.__split_idx:]
        self.chunks = split_dataset_to_chunks(self.data, days_before=config.days_before,
                                              day_prediction=config.days_prediction, chunk_spacing=chunk_spacing)

    def __getitem__(self, index):
        inputs, outputs = json.loads(json.dumps(self.chunks[index], indent=4))  # deep copy of json item
        features = []
        for input in inputs:
            out_item = dict()
            out_item['date'] = input['date']
            out_item['Price'] = input['Price']
            latest_news = input['latest_news']
            titles, contents = [], []
            if latest_news:
                titles, contents = list(zip(*[list(i.values()) for i in latest_news]))

            # convert to tensor
            titles = config.tokenizer(titles, truncation=True, padding='max_length', max_length=config.title_max_length,
                                      return_tensors='pt')['input_ids'] if titles else torch.tensor([0])
            contents = config.tokenizer(contents, truncation=True, padding='max_length', max_length=config.content_max_length,
                                      return_tensors='pt')['input_ids'] if contents else torch.tensor([0])
            out_item['titles'] = titles
            out_item['contents'] = contents
            features.append(out_item)

        features = [list(i.values()) for i in features]
        labels = [output['Price'] for output in outputs]  # label is the price and change %

        return features, labels

    def __len__(self):
        return len(self.chunks)


def collate_fn(batch):
    features, labels = list(zip(*batch))
    labels = torch.FloatTensor(labels)
    return features, labels


train_dataloader = DataLoader(NewsAndStocks(train=True), batch_size=config.train_batch_size,
                              collate_fn=collate_fn, shuffle=True, drop_last=True)
test_dataloader = DataLoader(NewsAndStocks(train=False, chunk_spacing=3), batch_size=config.train_batch_size,
                              collate_fn=collate_fn, shuffle=True, drop_last=True)

print('Dataloader loaded.')


if __name__ == '__main__':
    # create instance of dataset
    # dataset = NewsAndStocks()
    # print(len(dataset))
    # _features, _labels = dataset[0]
    # print(_labels)

    for item in test_dataloader:
        features, labels = item
        break

    print(features[0])

    # print(len(features), len(labels))  # batch_size
    # print(len(features[0]), len(labels[0]))  # each features and labels
    #
    # print(labels[0])
    #
    # # take change % only
    # print(labels[0][:, 1:])
    #
    # print(labels[:, :, 1:], labels[:, :, 1:].shape)

