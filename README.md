# insights-from-news
Prediction of stock price change given previous news data


# Concept
"Current and past news will cause effect on future stock price"

We try to let a model accept `previous news and previous stock price and change %` as input, and let the model predict `future stock price change %` as output.


# Model
Before the news are fed to model, text need to be tokenized. For this model, we use `pretrained bert tokenizer model` provided by HuggingFace.

Model utilize a seq2seq-like structure to stimulate the process of a human stock trader consume news data and predict future trends.

Encoder convert news titles and contents to `encoder_outputs` and `encoder_hidden`. The `encoder_hidden` will be passed to decoder to make prediction of future trend.

Decoder takes `encoder_hidden` and embedded `previous_change_perc` as the decoder rnn's hidden_state and input respectively. Output of decoder is the predicted trend.

# Result
The model gain a acceptable performance in the limited input data:

&nbsp;

<img src="output graphs\30 news to 14 prices 1.png" alt="prediction output">

<i>Prediction Outupt using test dataset</i>

&nbsp;

# conclusion
Model's performance is limited by the following conditions:
1. Number of input data. 
    - More input data fed, more pattern leanrt, more accurate result

2. Structure
    - Structure used(seq2seq-like) is a prototype for our concept, therefore it is rather simple. We may try attention machanism or transformer in the future, it may increase preformance and robustness of the model.