# 🌸 Streamlit 機械学習ダッシュボード

Streamlit を使った機械学習ダッシュボードのサンプルプロジェクト。
Iris データセットを使用した分類モデルの学習・評価・予測を行います。

## 📖 詳細記事

詳しい実装手順は以下の記事をご覧ください：

- [Streamlit で機械学習ダッシュボードを作ってみた - Peaky AI LAB](https://peaky.co.jp/streamlit-ml-dashboard-tutorial)

## ✨ 機能

- **データ探索**: 統計情報、散布図、ヒストグラム、相関行列
- **モデル学習**: パラメータ調整、ランダムフォレスト学習
- **予測機能**: 新しいデータからの品種予測
- **モデル評価**: 精度、混同行列、品種ごとの評価

## 🚀 クイックスタート

### 必要要件

- Python 3.8 以上

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/streamlit-ml-dashboard.git
cd streamlit-ml-dashboard

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 実行

```bash
streamlit run app.py
```

ブラウザが自動で開き、`http://localhost:8501`でアプリにアクセスできます。

## 📦 使用ライブラリ

- Streamlit 1.31.0
- scikit-learn 1.4.0
- pandas 2.1.4
- numpy 1.26.3
- plotly 5.18.0

## 📝 ライセンス

MIT License

## 🔗 関連リンク

- [Streamlit 公式ドキュメント](https://docs.streamlit.io/)
- [scikit-learn 公式ドキュメント](https://scikit-learn.org/)

```

### 2. requirements.txt
```

streamlit==1.31.0
scikit-learn==1.4.0
pandas==2.1.4
numpy==1.26.3
plotly==5.18.0
matplotlib==3.8.2
seaborn==0.13.1

```

### 3. .gitignore
```

# Python

**pycache**/
_.py[cod]
_$py.class
\*.so
.Python
env/
venv/
ENV/

# Streamlit

.streamlit/

# IDE

.vscode/
.idea/

# OS

.DS_Store
Thumbs.db

# Model files

_.pkl
_.joblib
