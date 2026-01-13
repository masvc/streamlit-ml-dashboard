import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

# ページ設定
st.set_page_config(
    page_title="アヤメ分類ダッシュボード",
    page_icon="🌸",
    layout="wide"
)

# タイトル
st.title('🌸 機械学習ダッシュボード: アヤメ品種分類')
st.markdown('Irisデータセットを使った分類モデルの学習・評価・予測')

# データ読み込み
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris

df, iris = load_data()

# サイドバー
st.sidebar.header('セクション選択')
section = st.sidebar.radio(
    'セクションを選んでください',
    ['データ探索', 'モデル学習', '予測', 'モデル評価']
)

# データ探索セクション
if section == 'データ探索':
    st.header('📊 データ探索')

    # データ表示
    st.subheader('データセット')
    st.dataframe(df.head(10))

    # 基本統計量
    st.subheader('基本統計量')
    st.write(df.describe())

    # データの分布を可視化
    st.subheader('データ分布')

    col1, col2 = st.columns(2)

    with col1:
        # 散布図
        feature_x = st.selectbox('X軸の特徴量', iris.feature_names, index=0)
        feature_y = st.selectbox('Y軸の特徴量', iris.feature_names, index=1)

        fig = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            color='species_name',
            title=f'{feature_x} vs {feature_y}',
            labels={'species_name': '品種'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ヒストグラム
        selected_feature = st.selectbox('特徴量を選択', iris.feature_names)

        fig = px.histogram(
            df,
            x=selected_feature,
            color='species_name',
            title=f'{selected_feature}の分布',
            labels={'species_name': '品種'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # 相関行列
    st.subheader('特徴量の相関')
    corr_matrix = df[iris.feature_names].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(title='相関行列')
    st.plotly_chart(fig, use_container_width=True)

# モデル学習セクション
elif section == 'モデル学習':
    st.header('🤖 モデル学習')
    st.markdown('ランダムフォレストのパラメータを調整してモデルを学習します')

    # パラメータ設定
    col1, col2, col3 = st.columns(3)

    with col1:
        n_estimators = st.slider('決定木の数', 10, 200, 100, 10)

    with col2:
        max_depth = st.slider('木の最大深さ', 1, 20, 10)

    with col3:
        test_size = st.slider('テストデータの割合', 0.1, 0.5, 0.2, 0.05)

    # データ分割
    X = df[iris.feature_names]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 学習実行
    if st.button('学習を開始', type='primary'):
        with st.spinner('学習中...'):
            # モデル作成
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            # 学習
            model.fit(X_train, y_train)

            # 予測
            y_pred = model.predict(X_test)

            # 精度計算
            accuracy = accuracy_score(y_test, y_pred)

            # セッションステートに保存
            st.session_state['model'] = model
            st.session_state['accuracy'] = accuracy
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred

        st.success('学習完了！')
        st.metric('テストデータの精度', f'{accuracy:.2%}')

        # 特徴量の重要度
        st.subheader('特徴量の重要度')
        feature_importance = pd.DataFrame({
            '特徴量': iris.feature_names,
            '重要度': model.feature_importances_
        }).sort_values('重要度', ascending=False)

        fig = px.bar(
            feature_importance,
            x='重要度',
            y='特徴量',
            orientation='h',
            title='特徴量の重要度'
        )
        st.plotly_chart(fig, use_container_width=True)

    # データ分割の可視化
    st.subheader('データ分割')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('訓練データ', f'{len(X_train)} サンプル')
    with col2:
        st.metric('テストデータ', f'{len(X_test)} サンプル')

# 予測セクション
elif section == '予測':
    st.header('🔮 品種予測')

    # モデルが学習済みかチェック
    if 'model' not in st.session_state:
        st.warning('先に「モデル学習」セクションでモデルを学習してください')
    else:
        st.markdown('各特徴量の値を入力して、アヤメの品種を予測します')

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                'がく片の長さ (cm)',
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )
            sepal_width = st.number_input(
                'がく片の幅 (cm)',
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.1
            )

        with col2:
            petal_length = st.number_input(
                '花びらの長さ (cm)',
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.1
            )
            petal_width = st.number_input(
                '花びらの幅 (cm)',
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1
            )

        # 予測実行
        if st.button('予測を実行', type='primary'):
            model = st.session_state['model']

            # 入力データを作成
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # 予測
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # 品種名
            species_names = ['setosa', 'versicolor', 'virginica']
            predicted_species = species_names[prediction]

            # 結果表示
            st.success(f'予測結果: **{predicted_species}**')

            # 確信度を表示
            st.subheader('各品種の確率')
            proba_df = pd.DataFrame({
                '品種': species_names,
                '確率': prediction_proba
            })

            fig = px.bar(
                proba_df,
                x='品種',
                y='確率',
                title='予測確率',
                range_y=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

            # 入力値の可視化
            st.subheader('入力した特徴量')
            input_df = pd.DataFrame({
                '特徴量': iris.feature_names,
                '値': [sepal_length, sepal_width, petal_length, petal_width]
            })

            fig = px.bar(
                input_df,
                x='特徴量',
                y='値',
                title='入力データ'
            )
            st.plotly_chart(fig, use_container_width=True)

# モデル評価セクション
elif section == 'モデル評価':
    st.header('📈 モデル評価')

    # モデルが学習済みかチェック
    if 'model' not in st.session_state:
        st.warning('先に「モデル学習」セクションでモデルを学習してください')
    else:
        accuracy = st.session_state['accuracy']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']

        # 精度表示
        st.metric('テストデータの精度', f'{accuracy:.2%}')

        # 混同行列
        st.subheader('混同行列')
        cm = confusion_matrix(y_test, y_pred)

        species_names = ['setosa', 'versicolor', 'virginica']

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=species_names,
            y=species_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={'size': 16}
        ))
        fig.update_layout(
            title='混同行列',
            xaxis_title='予測ラベル',
            yaxis_title='真のラベル'
        )
        st.plotly_chart(fig, use_container_width=True)

        # 品種ごとの精度
        st.subheader('品種ごとの詳細')

        report = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)

        report_df = pd.DataFrame(report).transpose()[:-3]  # micro/macro avgを除外
        report_df = report_df.reset_index()
        report_df.columns = ['品種', '適合率', '再現率', 'F1スコア', 'サポート']

        st.dataframe(report_df, use_container_width=True)

        # 予測結果の分布
        st.subheader('予測結果の分布')

        comparison_df = pd.DataFrame({
            '真のラベル': [species_names[i] for i in y_test],
            '予測ラベル': [species_names[i] for i in y_pred]
        })

        fig = px.scatter(
            comparison_df,
            x='真のラベル',
            y='予測ラベル',
            title='真のラベル vs 予測ラベル'
        )
        st.plotly_chart(fig, use_container_width=True)
