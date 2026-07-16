# ヒストグラム分析ツール（改善版）

Streamlit Community Cloudで公開できるヒストグラム分析アプリです。

## 主な機能

- Excel（.xlsx）の複数シート・複数列に対応
- 空欄、非数値、無限大の件数を表示
- Freedman–Diaconis、平方根則、スタージェス、Scottから階級幅を選択
- 列ごとの階級幅・開始位置の手動設定
- 度数、割合（%）、確率密度の切り替え
- 平均、中央値、標本標準偏差、四分位数などを表示
- 表で選んだ値をヒストグラム上に表示
- 共通階級による正しい重ね比較
- グラフPNG、統計CSV、分析結果Excelのダウンロード

## GitHub上の既存アプリを更新する方法

1. 既存リポジトリの `app.py` を、このフォルダの `app.py` に置き換える
2. `requirements.txt` と `packages.txt` も置き換える
3. Commit changesを押す
4. Streamlit Community Cloudが自動で再デプロイする

`streamlit-aggrid`は使用しなくなったため、requirements.txtから削除されています。

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
```
