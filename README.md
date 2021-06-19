# prediction

株価予測ツール　/ main.ipynb

言語:Python

ライブラリ:torch/matplotlib

概要
①Yahoo Finance から過去の株価情報を取得（symbols_names関数に、株のシンボルを入力してそれから取得）し、株価データを正規化
②株価データをLSTMモデルに学習させて、未来のデータを予測　
③生成された未来のデータを正規化の大きさから、株価の大きさに変換
④予測された株価データをmatplotlibを使ってグラフ化
