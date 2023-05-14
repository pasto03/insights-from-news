import torch
import torch.nn as nn
import config


# hyperparameters
embedding_dim = 150
vocab_size = config.tokenizer.vocab_size
num_layers = 4
hidden_size = 64

device = config.device

title_max_length = config.title_max_length
content_max_length = config.content_max_length


# create encoder class
class NewsEncoder(nn.Module):
    def __init__(self, out_dim=None):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.info_gru = nn.GRU(input_size=embedding_dim,
                               num_layers=num_layers,
                               hidden_size=hidden_size,
                               batch_first=True)
        self.comb_gru = nn.GRU(input_size=hidden_size,
                               num_layers=num_layers,
                               hidden_size=hidden_size,
                               batch_first=True)

    def forward(self, features):
        # 1. batch uneven embedding
        batch_embedded_titles = []
        batch_embedded_contents = []
        batch_input_prices = []
        for batch in features:
            dates, prices, titles, contents = list(zip(*batch))
            # print(prices)
            prices = torch.tensor(prices).to(device)
            titles = [i.to(device) for i in titles]
            contents = [i.to(device) for i in contents]
            embedded_titles = self.__uneven_embedding(titles, max_length=title_max_length)
            embedded_contents = self.__uneven_embedding(contents, max_length=content_max_length)
            batch_embedded_titles.append(embedded_titles)
            batch_embedded_contents.append(embedded_contents)
            batch_input_prices.append(prices)
        batch_embedded_titles = torch.stack(batch_embedded_titles)
        batch_embedded_contents = torch.stack(batch_embedded_contents)
        batch_input_prices = torch.stack(batch_input_prices)

        # 2. rnn
        out_titles = torch.stack([self.info_gru(embedded_titles)[0] for embedded_titles in batch_embedded_titles])
        out_contents = torch.stack(
            [self.info_gru(embedded_contents)[0] for embedded_contents in batch_embedded_contents])
        concat_info = torch.concat((out_titles, out_contents), dim=2)

        encoder_outputs, encoder_hidden = list(zip(*[self.comb_gru(c) for c in concat_info]))
        encoder_outputs, encoder_hidden = torch.stack(encoder_outputs), torch.stack(encoder_hidden)

        return encoder_outputs, encoder_hidden, batch_input_prices

    def __uneven_embedding(self, uneven_tlist, max_length):
        stacked_embedded = []
        for uneven in uneven_tlist:
            uneven = uneven.to(torch.int32).to(device)
            embedded = torch.zeros([max_length, embedding_dim]).to(device)
            if uneven.ndim == 2:
                for tensor in uneven:
                    embedded += self.embedding(tensor.to(device))
            stacked_embedded.append(embedded)

        stacked_embedded = torch.stack(stacked_embedded)
        return stacked_embedded  # [days_before, max_length, embedding_dim]


if __name__ == '__main__':
    from dataset import train_dataloader

    news_encoder = NewsEncoder().to(device)
    print(news_encoder)

    # test encoder
    for features, labels in train_dataloader:
        batch_encoder_outputs, batch_encoder_hidden, batch_input_prices = news_encoder(features)
        break

    print([batch_encoder_outputs.shape, batch_encoder_hidden.shape, batch_input_prices.shape])

