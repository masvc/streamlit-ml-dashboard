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
- [uv](https://docs.astral.sh/uv/) (推奨) または pip

### インストール

#### uv を使う場合（推奨・高速）

```bash
# リポジトリをクローン
git clone https://github.com/masvc/streamlit-ml-dashboard.git
cd streamlit-ml-dashboard

# uvがインストールされていない場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存パッケージをインストール
uv pip install -r requirements.txt

# アプリを実行
uv run streamlit run app.py
```

#### pip を使う場合

```bash
# リポジトリをクローン
git clone https://github.com/masvc/streamlit-ml-dashboard.git
cd streamlit-ml-dashboard

# 仮想環境を作成
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt

# アプリを実行
streamlit run app.py
```

ブラウザが自動で開き、`http://localhost:8501`でアプリにアクセスできます。

## 📦 使用ライブラリ

- Streamlit 1.31.0
- scikit-learn 1.4.0
- pandas 2.1.4
- numpy 1.26.3
- plotly 5.18.0
- matplotlib 3.8.2
- seaborn 0.13.1

## 📸 スクリーンショット

### データ探索

データセットの統計情報、散布図、ヒストグラム、相関行列を表示

### モデル学習

パラメータを調整してランダムフォレストを学習

### 予測

新しいアヤメのデータから品種を予測

### モデル評価

精度、混同行列、品種ごとの詳細評価を表示

## 🔧 開発

### uv を使った開発フロー

```bash
# 依存パッケージの追加
uv pip install <package-name>

# requirements.txtの更新
uv pip freeze > requirements.txt

# アプリの実行
uv run streamlit run app.py
```

## 📝 ライセンス

MIT License

## 🔗 関連リンク

- [Streamlit 公式ドキュメント](https://docs.streamlit.io/)
- [scikit-learn 公式ドキュメント](https://scikit-learn.org/)
- [uv - Python パッケージマネージャー](https://docs.astral.sh/uv/)
- [Peaky AI LAB - AI 開発ツール・技術記事](https://peaky.co.jp/)
