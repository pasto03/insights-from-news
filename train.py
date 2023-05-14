import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from seq2seq import Seq2Seq
import config
from dataset import train_dataloader

device = config.device
model = Seq2Seq().to(device)
optimizer = Adam(model.parameters(), lr=5e-3)

print('Model and optimizer completed setup.')

print('There are {} batches in dataloader.'.format(len(train_dataloader)))


def train():
    for batch_idx, (features, labels) in tqdm(enumerate(train_dataloader),
                                              total=len(train_dataloader), desc='train process'):
        output = model(features)
        output_perc = output[:, :, :1]
        output_signs = output[:, :, 1:]
        labels = labels.to(device)[:, :, 1:]   # take perc % for each batch
        # sign to be used in second loss calc
        label_signs = torch.where(labels >= 0, torch.tensor(1), torch.tensor(0)).to(torch.float32)

        # loss 1: mse
        mse_loss = F.mse_loss(output_perc, labels)
        # print('mse_loss:', mse_loss)

        # loss 2: bce
        bce_loss = nn.BCELoss()(output_signs, label_signs)
        # print('bce_loss:', bce_loss)

        loss = mse_loss + bce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % (len(train_dataloader) // 4) == 0:
            print(f'batch {batch_idx}: {loss.item()}')

    return loss.item(), output, labels


if __name__ == '__main__':
    for epoch in range(3):
        print('epoch {}:'.format(epoch))
        loss, output, labels = train()

    # visualize result
    import matplotlib.pyplot as plt
    _output = output[0][:, :1].flatten().tolist()
    _labels = labels[0].flatten().tolist()

    plt.title('14 days stock change prediction given 30 days previous news')
    plt.plot(_output, label='predict')
    plt.plot(_labels, label='real')
    plt.legend()
    plt.show()

save_choice = input('Save model?(Y/n) ')
if save_choice.lower() == 'y':
    torch.save(model.state_dict(), 'model/state_dict.pkl')

