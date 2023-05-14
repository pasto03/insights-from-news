import torch
import torch.nn as nn
from encoder import NewsEncoder
from decoder import NewsDecoder
# import config


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = NewsEncoder()
        self.decoder = NewsDecoder()

    def forward(self, features):
        _, batch_encoder_hidden, batch_input_prices = self.encoder(features)
        batch_output = self.decoder(batch_encoder_hidden, batch_input_prices)
        return batch_output


if __name__ == '__main__':
    from dataset import train_dataloader
    # test model
    model = Seq2Seq().to(device)

    for features, labels in train_dataloader:
        output = model(features)
        break

    print(output)
    print(output.shape)
    print(labels.shape)

