import torch
from seq2seq import Seq2Seq
from dataset import test_dataloader
import config
import matplotlib.pyplot as plt


device = config.device
model = Seq2Seq().to(device)
model.load_state_dict(torch.load('model/state_dict.pkl'))

model.eval()
with torch.no_grad():
    for features, labels in test_dataloader:
        output = model(features)
        output_perc = output[:, :, :1]
        output_signs = output[:, :, 1:]
        labels_perc = labels.to(device)[:, :, 1:]   # take perc % for each batch

        plt.figure(figsize=(30, 30))
        plt.suptitle('14 days stock change prediction given 30 days previous news')
        for i in range(24):
            plt.subplot(6, 4, i+1)
            # convert predict and real values to list
            _output_perc = output_perc[i].flatten().tolist()
            _labels_perc = labels_perc[i].flatten().tolist()

            # plt.title('14 days stock change prediction given 30 days previous news')
            plt.plot(_output_perc, label='predict')
            plt.plot(_labels_perc, label='real')
            plt.xlim(0, config.days_prediction+1)
            plt.legend()
        plt.show()

        break

    # show predicted and actual prices
    print(output[0])
    last_price = features[0][-1][1][0]
    print('last_price:', last_price)
    predicted_prices = []
    for _output in output[0]:
        _output_perc = _output[0]
        perc_change = _output_perc / 100
        price_change = last_price * perc_change
        predicted_price = last_price + price_change
        predicted_price = float(predicted_price.cpu().detach())
        predicted_prices.append(predicted_price)
        last_price = predicted_price
    print('Predicted prices:', predicted_prices)
    real_prices = [float(i) for i in labels[0, :, 0].cpu().detach()]
    print('Real prices:', real_prices)

