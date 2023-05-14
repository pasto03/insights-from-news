import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# hyperparameters
embedding_dim = 150
vocab_size = config.tokenizer.vocab_size
num_layers = 4
hidden_size = 64

device = config.device

days_before = config.days_before
days_prediction = config.days_prediction


class NewsDecoder(nn.Module):
    def __init__(self):
        super(NewsDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim,
                          num_layers=num_layers,
                          hidden_size=hidden_size,
                          batch_first=True)
        self.fc = nn.Linear(days_before * hidden_size * 2, days_prediction)

    def forward(self, batch_encoder_hidden, batch_input_prices):
        batch_size = batch_input_prices.shape[0]
        batch_output = []
        for batch_idx in range(batch_size):
            encoder_hidden = batch_encoder_hidden[batch_idx]
            in_change_perc = batch_input_prices[batch_idx]
            in_change_perc = in_change_perc[:, 1:].to(torch.int64)
            # embedding only accepts positive integers
            perc_signs = torch.where(in_change_perc >= 0, torch.tensor(1), torch.tensor(0))
            perc_concat = torch.concat((abs(in_change_perc), perc_signs), dim=-1)
            embedded = self.embedding(perc_concat)
            out, _ = self.gru(embedded, encoder_hidden)
            out = out.flatten()
            output = self.fc(out)
            batch_output.append(output)
        batch_output = torch.stack(batch_output).unsqueeze(-1)
        output_signs = torch.where(batch_output >= 0, torch.tensor(1), torch.tensor(0))
        concat_output = torch.concat((batch_output, output_signs), dim=-1)
        return concat_output


if __name__ == '__main__':
    from encoder import NewsEncoder
    from dataset import train_dataloader

    news_decoder = NewsDecoder().to(device)
    print(news_decoder)

    # get encoder and its output
    news_encoder = NewsEncoder().to(device)

    # test encoder
    for features, labels in train_dataloader:
        batch_encoder_outputs, batch_encoder_hidden, batch_input_prices = news_encoder(features)
        break

    print(batch_input_prices.shape)

    # try decoder
    batch_output = news_decoder(batch_encoder_hidden, batch_input_prices)
    print(batch_output.shape)

