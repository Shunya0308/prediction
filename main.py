import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler
import pandas.tseries.offsets as offsets

# 日立 Hitachi, Ltd.  (6501.JP)
stock_data = data.DataReader('6501.JP', 'stooq').sort_values('Date', ascending=True)

# 東芝 Toshiba Co. (6502.JP)
stock_data = data.DataReader('6502.JP', 'stooq').sort_values('Date', ascending=True)

# サイボウズ Cybozu, Inc. (4776.JP)
stock_data = data.DataReader('4776.JP', 'stooq').sort_values('Date', ascending=True)

# 弁護士ドットコム Bengo4.com, Inc. (6027.JP)
stock_data = data.DataReader('6027.JP', 'stooq').sort_values('Date', ascending=True)

# メルカリ Mercari, Inc. (4385.JP)
stock_data = data.DataReader('4385.JP', 'stooq').sort_values('Date', ascending=True)

stock_data = stock_data.drop(["Open", "High", "Low", "Volume"], axis=1)

stock_data["Close"]


# データの数値を-1から1の間に収める。 (データの正規化)
y = stock_data["Close"].values
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(y.reshape(-1, 1))
y = scaler.transform(y.reshape(-1, 1))
y = torch.FloatTensor(y).view(-1)


plt.figure(figsize=(12, 4))
plt.xlim(-20, len(y)+20)
plt.grid(True)
plt.plot(y)


train_window_size = 7

def input_data(seq, ws):
    out = []
    L = len(seq)

    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))

    return out

# 直前までの全てのデータを、トレーニング用として、モデルに渡す
train_data = input_data(y, train_window_size)


class Model(nn.Module):

    def __init__(self, input=1, h=50, output=1):
        super().__init__()
        self.hidden_size = h

        self.lstm = nn.LSTM(input, h)
        self.fc = nn.Linear(h, output)

        self.hidden = (
            torch.zeros(1, 1, h),
            torch.zeros(1, 1, h)
        )

    def forword(self, seq):

        out, _ = self.lstm(
            seq.view(len(seq), 1, -1),
            self.hidden
        )

        out = self.fc(
            out.view(len(seq), -1)
        )

        return out[-1]


torch.manual_seed(123)
model = Model()

# mean squared error loss 平均二乗誤差法
criterion = nn.MSELoss()

# stocastic gradient descent 確率的勾配降下法
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def run_train():
    for train_window, correct_label in train_data:

        optimizer.zero_grad()

        model.hidden = (
            torch.zeros(1, 1, model.hidden_size),
            torch.zeros(1, 1, model.hidden_size)
        )

        train_predicted_label = model.forword(train_window)
        train_loss = criterion(train_predicted_label, correct_label)

        train_loss.backward()
        optimizer.step()


test_size = 30


def run_test():
    for i in range(test_size):
        test_window = torch.FloatTensor(extending_seq[-test_size:])

        with torch.no_grad():
            model.hidden = (
                torch.zeros(1, 1, model.hidden_size),
                torch.zeros(1, 1, model.hidden_size)
            )

            test_predicted_label = model.forword(test_window)
            extending_seq.append(test_predicted_label.item())


epochs = 20

for epoch in range(epochs):

    print()
    print(f'Epoch: {epoch+1}')

    run_train()

    extending_seq = y[-test_size:].tolist()

    run_test()

    plt.figure(figsize=(12, 4))
    plt.xlim(-20, len(y)+50)
    plt.grid(True)

    plt.plot(y)

    plt.plot(
        range(len(y), len(y)+test_size),
        extending_seq[-test_size:]
    )

    plt.show()


# 未来のデータの数値を、株価のスケールに変換
predicted_nomalized_labels_list = extending_seq[-test_size:]
predicted_nomalized_labels_array_1d = np.array(predicted_nomalized_labels_list)
predicted_nomalized_labels_array_2d = predicted_nomalized_labels_array_1d.reshape(-1, 1)
predicted_labels_array_2d = scaler.inverse_transform(predicted_nomalized_labels_array_2d)


plt.figure(figsize=(12, 4))
plt.xlim(-20, len(predicted_labels_array_2d)+20)
plt.grid(True)
plt.plot(predicted_labels_array_2d)


# 未来３０日間の日付を取得
np.arange('2021-01-01', '2021-02-01', dtype='datetime64')


# 直近のデータの最後の日付 （Timestamp型）
real_last_date_timestamp = stock_data.index[-1]


# 未来の最初の日付　（Timestamp型）
future_first_date_timestamp = real_last_date_timestamp + offsets.Day()


# 未来の最初の日付　（Series datetime型）
future_first_date_series_datetime = pd.Series(future_first_date_timestamp)


# 未来の最初の日付　（Series object型）
future_first_date_series_object = pd.Series(future_first_date_timestamp).astype(str)


# 未来の最初の日付（str型）
future_first_date_str = future_first_date_series_object[0]


# 未来の最初の日付の翌日の日付（Timestamp型）
second_argument_timestamp = future_first_date_timestamp + offsets.Day(30)


# 未来の最初の日付の翌日の日付（Series　datetime型）
second_argument_series_datetime = pd.Series(second_argument_timestamp)


# 未来の最初の日付　（Series object型）
second_argument_series_object = pd.Series(second_argument_timestamp).astype(str)


# 未来の最初の日付（str型）
second_argument_str = second_argument_series_object[0]


# 直近の全てのデータを学習用に使う場合（test_size = 30）
future_period = np.arange(future_first_date_str, second_argument_str, dtype='datetime64')


# 直近のデータの最後の日付から３ヶ月ほど、遡った日付（Timestamp型）
plot_start_date_timestamp = real_last_date_timestamp + offsets.Day(-90)


fig = plt.figure(figsize=(12, 4))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

#plt.plot(stock_data["Close"][plot_start_date_timestamp:], label="Hitachi")
plt.plot(stock_data["Close"][plot_start_date_timestamp:], label="Toshiba")
#plt.plot(stock_data["Close"][plot_start_date_timestamp:], label="Cybozu")
#plt.plot(stock_data["Close"][plot_start_date_timestamp:], label="Bengo4.com")
#plt.plot(stock_data["Close"][plot_start_date_timestamp:], label="Mercari")



plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=18)

plt.plot(future_period, predicted_labels_array_2d)

plt.show()