import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import streamlit as st

# Configurar o estilo do Streamlit
st.title("Previsão de Carga usando LSTM")

# Carregar os dados
st.write("### Carregando os dados...")
uploaded_file = st.file_uploader("Carregue o arquivo Excel com os dados:", type=["xlsx"])
if uploaded_file is not None:
    dados = pd.read_excel(uploaded_file)
    st.write("Visualização dos dados:")
    st.dataframe(dados.head())
    
    # Extraindo dados
    carga = dados.iloc[:20000, 3].values  # Coluna 4 (índice 3)
    
    # Normalizar os dados
    scaler = StandardScaler()
    carga_normalizada = scaler.fit_transform(carga.reshape(-1, 1)).flatten()
    
    # Criar dados em formato de série temporal
    def criar_dados(series, passos=50):
        X, Y = [], []
        for i in range(len(series) - passos):
            X.append(series[i:i + passos])
            Y.append(series[i + passos])
        return np.array(X), np.array(Y)

    # Preparar os dados
    passos_anteriores = 75
    X, Y = criar_dados(carga_normalizada, passos_anteriores)

    # Dividir os dados em treino e teste
    split = int(0.6 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    # Ajustar o formato para LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Criar o modelo LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(75, activation='tanh', input_shape=(passos_anteriores, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    st.write("Treinando o modelo...")
    # Treinar o modelo
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=75,
        batch_size=132,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Fazer previsões
    previsoes = model.predict(X_test)
    previsoes = scaler.inverse_transform(previsoes)
    Y_test_real = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Avaliar o modelo
    MAPE = np.mean(np.abs((previsoes - Y_test_real) / Y_test_real)) * 100
    MAE = np.mean(np.abs(previsoes - Y_test_real))
    RMSE = np.sqrt(np.mean((previsoes - Y_test_real) ** 2))

    st.write(f"**Erro Percentual Absoluto Médio (MAPE):** {MAPE:.4f}%")
    st.write(f"**Erro Absoluto Médio (MAE):** {MAE:.4f}")
    st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** {RMSE:.4f}")

    # Criar o gráfico interativo com Plotly
    st.write("### Visualização dos Resultados Interativa:")
    fig = go.Figure()

    # Adicionar dados reais
    fig.add_trace(go.Scatter(
        y=Y_test_real.flatten(),
        mode='lines',
        name='Dados Reais',
        line=dict(color='blue')
    ))

    # Adicionar previsões
    fig.add_trace(go.Scatter(
        y=previsoes.flatten(),
        mode='lines',
        name='Previsão',
        line=dict(color='red', dash='dash')
    ))

    # Configurar o layout
    fig.update_layout(
        title="Previsão de Carga usando LSTM",
        xaxis_title="Tempo",
        yaxis_title="Carga",
        legend=dict(x=0, y=1),
        template="plotly_white"
    )

    # Mostrar o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)
