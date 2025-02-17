import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import base64
import requests
from io import BytesIO
from PIL import Image
st.set_page_config(layout="wide")
# Aplica o tema claro manualmente
st.markdown(
    """
    <style>
        body {
            background-color: white !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        /* Fundo principal (modo Light padrão) */
        .stApp {
            background-color: #FFFFFF !important;
        }

        /* Texto padrão */
        h1, h2, h3, h4, h5, h6, p, label, div {
            color: #000000 !important;
        }

        /* Fundo da Sidebar */
        [data-testid="stSidebar"] {
            background-color: #9AD6CC !important;
        }

        /* Botões e elementos interativos */
        .stButton>button {
            background-color: #FFFFFF !important;
            color: #FFFFFF !important;
            border: 1px solid #CCCCCC !important;
        }

        .reportview-container {
            background-color: #fFfffF;
        }
        /* Alteração de links */
        a {
            color: #1E70BF !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def CriarGraficos(fig, graph_num):
    # Cria o diretório 'images' se ele não existir
    if not os.path.exists("images"):
        os.makedirs("images")

    # Cria um identificador único para os widgets deste gráfico no Session State
    widget_id = f"graph_{graph_num}"

    # Nomes dos arquivos de saída
    output_file_jpeg = f"images/grafico_{graph_num}_high_quality.jpg"
    output_file_png = f"images/grafico_{graph_num}_high_quality.png"
    output_file_svg = f"images/grafico_{graph_num}_high_quality.svg"

    # Salvando as imagens nos diferentes formatos
    fig.write_image(output_file_jpeg, format='jpeg', scale=3)
    fig.write_image(output_file_png, format='png', scale=3)
    fig.write_image(output_file_svg, format='svg', scale=3)

    # Inicializa o estado dos botões no Session State, se ainda não existirem
    if f"downloaded_jpeg_{widget_id}" not in st.session_state:
        st.session_state[f"downloaded_jpeg_{widget_id}"] = False
    if f"downloaded_png_{widget_id}" not in st.session_state:
        st.session_state[f"downloaded_png_{widget_id}"] = False
    if f"downloaded_svg_{widget_id}" not in st.session_state:
        st.session_state[f"downloaded_svg_{widget_id}"] = False

    # Criando colunas para os botões de download
    left, middle, right = st.columns(3)
    # Botão de download para JPEG
    with open(output_file_jpeg, "rb") as file:
        if left.download_button(
                label=f"Baixar Gráfico {graph_num} em JPEG",
                data=file,
                file_name=output_file_jpeg,
                mime="image/jpeg",
                key=f"download_jpeg_{widget_id}",  # Chave única
                # on_click=set_downloaded_state,
                #args=(f"downloaded_jpeg_{widget_id}",),
        ):
            pass  # Aqui você pode adicionar alguma lógica se necessário, após o clique

    # Botão de download para PNG
    with open(output_file_png, "rb") as file:
        if middle.download_button(
                label=f"Baixar Gráfico {graph_num} em PNG",
                data=file,
                file_name=output_file_png,
                mime="image/png",
                key=f"download_png_{widget_id}",  # Chave única
                # on_click=set_downloaded_state,
                # args=(f"downloaded_png_{widget_id}",),
        ):
            pass

    # Botão de download para SVG
    with open(output_file_svg, "rb") as file:
        if right.download_button(
                label=f"Baixar Gráfico {graph_num} em SVG",
                data=file,
                file_name=output_file_svg,
                mime="image/svg+xml",
                key=f"download_svg_{widget_id}",  # Chave única
                # on_click=set_downloaded_state,
                # args=(f"downloaded_svg_{widget_id}",),
        ):
            pass


def executar_codigo(dados, preco_solar, preco_eolica, preco_h2, preco_bateria, dolar_hoje, Potenciadosistema, epochs, batch_size, learning_rate, num_neurons, activation, passos_anteriores):
    # Carregando os dados
    carga = dados.iloc[10:1010, 14].values  # Coluna 4 (índice 3)
    scaler = StandardScaler()
    carga_normalizada = scaler.fit_transform(carga.reshape(-1, 1)).flatten()

    # Criar dados em formato de série temporal
    def criar_dados(series, passos=passos_anteriores):
        X, Y = [], []
        for i in range(len(series) - passos):
            X.append(series[i:i + passos])
            Y.append(series[i + passos])
        return np.array(X), np.array(Y)

    # Preparar os dados
    passos_anteriores = passos_anteriores  # a variável passos_anteriores (ou passos) define o número de passos no tempo (ou lags)
    # que o modelo LSTM usará como entrada para prever o próximo valor. Em outras palavras, ela indica quantos pontos de dados passados serão considerados como contexto para prever o valor futuro.
    X, Y = criar_dados(carga_normalizada, passos_anteriores)

    # Dividir os dados em treino e teste
    split = int(0.6 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    # Ajustar o formato para LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Criar um Dataset para treinar eficientemente o modelo LSTM usando o TensorFlow
    batch_size = batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    #  Criar o modelo LSTM com regularização L2
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(passos_anteriores, 1)),
        tf.keras.layers.LSTM(
            num_neurons,
            activation=activation,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),  # Regularização
            recurrent_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.LSTM(num_neurons, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compilar o modelo com a função de perda e otimizador
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Redução da taxa de aprendizado
        loss='mse'
    )
    model.summary()

    # % Configurar callbacks para o treinamento
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_loss', save_best_only=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    )

    # % Treinar o modelo
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
    )

    # % Fazer previsões
    previsoes = model.predict(X_test, batch_size=batch_size)
    previsoes = scaler.inverse_transform(previsoes)
    Y_test_real = scaler.inverse_transform(Y_test.reshape(-1, 1))
    carga_prevista = previsoes
    # % Avaliar o modelo
    MAPE = np.mean(np.abs((previsoes - Y_test_real) / Y_test_real)) * 100
    MAE = np.mean(np.abs(previsoes - Y_test_real))
    RMSE = np.sqrt(np.mean((previsoes - Y_test_real) ** 2))

    # print(f'Erro Percentual Absoluto Médio (MAPE): {MAPE:.4f}%')
    # print(f'Erro Absoluto Médio (MAE): {MAE:.4f}')
    # print(f'Raiz do Erro Quadrático Médio (RMSE): {RMSE:.4f}')

    # Título da aplicação
    st.title('Previsão de Carga usando LSTM')

    # Exibir métricas
    st.subheader('Métricas de Desempenho')
    st.write(f"**Erro Percentual Absoluto Médio (MAPE):** {MAPE:.4f}%")
    st.write(f"**Erro Absoluto Médio (MAE):** {MAE:.4f}")
    st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** {RMSE:.4f}")

    timesteps = list(range(1, len(Y_test_real) + 1))  # Substitua pelos índices ou tempos reais
    timesteps = list(timesteps)

    # Criar o gráfico interativo
    
    fig = go.Figure()
    # Adicionar linha dos dados reais
    fig.add_trace(go.Scatter(y=Y_test_real.flatten(), mode='lines', name='Dados Reais', line=dict(color='blue')
                             ))
    # Adicionar linha das previsões
    fig.add_trace(go.Scatter(y=previsoes.flatten(), mode='lines', name='Previsão', line=dict(color='red', dash='dash')
                             ))
    # Configurar layout do gráfico
    fig.update_layout(xaxis_title='Tempo em horas', yaxis_title='Carga (W)',
                      legend=dict(x=0, y=1),
                       width=1500, height=500, template='plotly_white',
                      )
    # Exibir no Streamlit
    st.title("Visualização de Previsão de Carga")
    st.subheader("Previsão de Carga usando LSTM")
    st.plotly_chart(fig)
    CriarGraficos(fig, 0)

    st.title("Despacho Econômico")

    ######################  despacho economico #########################

    # Configurações do Streamlit
    st.title("Análise de Otimização de Energia")
    st.subheader("Simulação de uma Microrrede com geração solar, eólica, H2 e bateria")

    # Leitura dos dados
    geracao = dados

    # Parâmetros iniciais
    G_Solar = np.zeros(288)
    Radiacao = np.array(geracao.iloc[1:289, 7])
    Temperatura = np.array(geracao.iloc[1:289, 6])
    Vento = np.array(geracao.iloc[1:289, 4])
    
    # Cálculo da geração solar
    for i in range(len(Radiacao)):
        G_Solar[i] = 0.97 * (Radiacao[i]) / 1000 * (1 + 0.005 * ((Temperatura[i]) - 25))

    # Parâmetros eólicos
    V_min = 3
    V_nominal = 12
    V_max = 25
    n_WT = 1
    eta_WT = 0.9
    P_R_WT = 1
    u_cut_in = 3
    u_rated = 12
    u_cut_off = 25

    # Potência gerada pelo vento
    P_WT = np.zeros(len(Vento))
    for i, u in enumerate(Vento):
        if u < u_cut_in:
            P_WT[i] = 0
        elif u_cut_in <= u <= u_rated:
            P_WT[i] = n_WT * eta_WT * P_R_WT * ((u ** 2 - u_cut_in ** 2) / (u_rated ** 2 - u_cut_in ** 2))
        elif u_rated < u < u_cut_off:
            P_WT[i] = n_WT * eta_WT * P_R_WT
        else:
            P_WT[i] = 0

    # Parâmetros gerais
    potenciadosistema = Potenciadosistema
    potenciaEolicamaxima_values = P_WT * potenciadosistema
    potenciasolarmaxima_values = G_Solar * potenciadosistema

    # Carga_Residencial = np.array(geracao.iloc[18:306, 10]) * Carga

    Carga_Residencial = np.array(carga_prevista[18:306])

    horas_uso_cel_comb = 8760
    vida_util_sistema = vida_util
    dolar_hoje = dolar_hoje
    precoH2 = preco_h2  # 14.3
    precoeolica = preco_eolica  # 0.1712
    precosolar = preco_solar  # 0.17566
    precobateria = preco_bateria  # 0.01
    tanque_hidrogenio_max = 45  # kg
    nivel_tanque = tanque_hidrogenio_max
    consumo_kg_h2 = 0.2  # consumo de hidrogenio em kg para produzir 1kWh
    eficiencia_eletrolisador = 0.75
    consumo_kW_h2 = -53571 * eficiencia_eletrolisador + 92643  # consumo de kW para produzir 1 kg de hidrogenio
    potencia_maxima_eletrolisador = Potenciadosistema
    niveis_eletrolisador = 6
    cel_combustivel = 20000
    # Possibilidades do eletrólise
    for n in range(1, niveis_eletrolisador + 1):
        if n == 1:
            possibilidades_eletrolisador = [0]
        else:
            possibilidades_eletrolisador = [0] + [i * potencia_maxima_eletrolisador / niveis_eletrolisador for i in
                                                  range(1, niveis_eletrolisador + 1)]

    # Inicialização
    pbat = 30000
    potenciaBateria_max = 27000
    potenciaH2maxima = 30000
    potenciaBateria_min = 3000
    status_bateria = np.zeros(len(Carga_Residencial))
    status_tanque = np.zeros(len(Carga_Residencial))

    # Inicializando listas para armazenar os resultados
    PotenciaSolar = []
    PotenciaEolica = []
    CustoTotal = []
    PotenciaH2 = []
    PotenciaBateria = []

    for i in range(len(potenciasolarmaxima_values)):
        status_bateria[i] = potenciaBateria_max
        status_tanque[i] = nivel_tanque
        potenciasolarmaxima = potenciasolarmaxima_values[i]
        potenciaEolicamaxima = potenciaEolicamaxima_values[i]

        ciclo = vida_util_sistema/20
        if ciclo > 1:
            # Custo eólica = preço do aerogerador + preço do inversor + preço da O&M do aerogerador + preço da O&M inversor
            preco_eolica = dolar_hoje * ((potenciaEolicamaxima / 1000) * (1000 )  + vida_util_sistema * (precoeolica ) + ciclo * 1000 )
            # preco_eolica =precoeolica
            # Custo solar = preço do painél fotovoltaico + preço do inversor + preço da O&M do painél fotovoltaico + preço da O&M inversor
            preco_solar = dolar_hoje * (
                        (potenciasolarmaxima / 1000) * (2000 + 625  )  + vida_util_sistema * (precosolar + 10) + ciclo * 2000 + ciclo * 625 )
            # preco_solar = precosolar
            # Custo bateria = preço das baterias + preço da O&M das baterias
            preco_bateria = dolar_hoje * (
                        (potenciaBateria_max * 200 / 1000)  + vida_util_sistema * precobateria + ciclo * 180 )
            # preco_bateria = precobateria
            # Custo da célula combustível = preço da célula combustível + preço da O&M das células
            preco_cel_comb = dolar_hoje * ((cel_combustivel * 3000 / 1000) + vida_util_sistema * 0.02 * horas_uso_cel_comb + ciclo * 2500 )

            # Custo do eletrolisador  = preço inicial do eletrolisador + preço da O&M do eletrolisador
            preco_eletrolisador = dolar_hoje * (
                        (potencia_maxima_eletrolisador * 1500 / 1000) + vida_util_sistema * precoH2 + ciclo * 1500 )

            # Custo do tanque = preço inicial do tanque + preço de O&M do tanque
            preco_tanque = dolar_hoje * nivel_tanque * (625 + vida_util_sistema * 10) + ciclo * 400*nivel_tanque  
            PrecoH2 = (preco_cel_comb + preco_eletrolisador + preco_tanque)
        else:
            # Custo eólica = preço do aerogerador + preço do inversor + preço da O&M do aerogerador + preço da O&M inversor
            preco_eolica = dolar_hoje * ((potenciaEolicamaxima / 1000) * (1000 )  + vida_util_sistema * (precoeolica ))
            # preco_eolica =precoeolica
            # Custo solar = preço do painél fotovoltaico + preço do inversor + preço da O&M do painél fotovoltaico + preço da O&M inversor
            preco_solar = dolar_hoje * (
                        (potenciasolarmaxima / 1000) * (2000 + 625  )  + vida_util_sistema * (precosolar + 10))
            # preco_solar = precosolar
            # Custo bateria = preço das baterias + preço da O&M das baterias
            preco_bateria = dolar_hoje * (
                        (potenciaBateria_max * 200 / 1000)  + vida_util_sistema * precobateria)
            # preco_bateria = precobateria
            # Custo da célula combustível = preço da célula combustível + preço da O&M das células
            preco_cel_comb = dolar_hoje * ((cel_combustivel * 3000 / 1000) + vida_util_sistema * 0.02 * horas_uso_cel_comb)

            # Custo do eletrolisador  = preço inicial do eletrolisador + preço da O&M do eletrolisador
            preco_eletrolisador = dolar_hoje * (
                        (potencia_maxima_eletrolisador * 1500 / 1000) + vida_util_sistema * precoH2)

            # Custo do tanque = preço inicial do tanque + preço de O&M do tanque
            preco_tanque = dolar_hoje * nivel_tanque * (625 + vida_util_sistema * 10)
            PrecoH2 = (preco_cel_comb + preco_eletrolisador + preco_tanque)
            # PrecoH2 = precoH2

        potencia_disponivel = potenciasolarmaxima + potenciaEolicamaxima

        # Limite mínimo para uso da bateria (10% da capacidade máxima)
        limite_minimo_bateria = 0.1 * pbat

        if Carga_Residencial[i][0] <= potencia_disponivel:
            # Problema de otimização sem bateria
            c = [preco_solar, preco_eolica, 0, 0]  # O custo da bateria é 0 (não usada)
            A_eq = [[1, 1, 0, 0]]  # Bateria excluída do problema
            b_eq = [Carga_Residencial[i][0]]
            bounds = [
                (0, potenciasolarmaxima),
                (0, potenciaEolicamaxima),
                (0, 0),
                (0, 0),  # Bateria com limite 0
            ]
        elif Carga_Residencial[i][
            0] <= potencia_disponivel + potenciaBateria_max and potenciaBateria_max > limite_minimo_bateria:
            c = [preco_solar, preco_eolica, 0, preco_bateria]
            A_eq = [[1, 1, 0, 1]]
            b_eq = [Carga_Residencial[i][0]]
            bounds = [
                (0, potenciasolarmaxima),
                (0, potenciaEolicamaxima),
                (0, 0),
                (limite_minimo_bateria + 1, potenciaBateria_max),
            ]
        else:
            # Geração insuficiente, incluir bateria no problema
            c = [preco_solar, preco_eolica, PrecoH2, 0]
            A_eq = [[1, 1, 1, 0]]
            b_eq = [Carga_Residencial[i][0]]
            bounds = [
                (0, potenciasolarmaxima),
                (0, potenciaEolicamaxima),
                (0, potenciaH2maxima),
                (0, 0),  # Bateria usada somente se acima de 10% de carga
            ]

        c = np.array(c).flatten()
        # Resolvendo o problema de otimização
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        PotenciaBateria.append(result.x[3])
        PotenciaSolar.append(result.x[0])
        PotenciaEolica.append(result.x[1])
        CustoTotal.append(result.fun)  # Adicionando o valor da função objetivo
        PotenciaH2.append(result.x[2])

        possivel_carga_bateria = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i][0]
        possivel_carga_H2 = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i][
            0] - possivel_carga_bateria
        if possivel_carga_bateria < 0:
            possivel_carga_bateria = 0
        if possivel_carga_H2 < 0:
            possivel_carga_H2 = 0
        if possivel_carga_bateria > pbat - potenciaBateria_max:
            possivel_carga_bateria = pbat - potenciaBateria_max  # igual ao valor da diferenca entre a capacidade maxima e a capacidade atual

        if nivel_tanque >= tanque_hidrogenio_max:
            possivel_carga_H2 = 0
        else:
            for n in range(2, niveis_eletrolisador + 2):
                if possivel_carga_H2 < possibilidades_eletrolisador[n - 1]:
                    possivel_carga_H2 = possibilidades_eletrolisador[n - 2]
                if n == (niveis_eletrolisador + 1) and possivel_carga_H2 >= possibilidades_eletrolisador[n - 1]:
                    possivel_carga_H2 = possibilidades_eletrolisador[n - 1]
        if tanque_hidrogenio_max - nivel_tanque < possivel_carga_H2 / consumo_kW_h2:
            possivel_carga_H2 = ((tanque_hidrogenio_max - nivel_tanque) * consumo_kW_h2)

        if result.success:
            x = result.x
            potenciaBateria_max = potenciaBateria_max - x[3] + possivel_carga_bateria
            # potenciaBateria_max1 = max(potenciaBateria_max, 3000)
            nivel_tanque = (nivel_tanque - (x[2] / 1000) * consumo_kg_h2 + (possivel_carga_H2 / consumo_kW_h2))
        if potenciaBateria_max < 3000:
            potenciaBateria_max = 3000

    PotenciaTotal = []
    for i in range(len(PotenciaEolica)):
        PotenciaTotal.append(potenciasolarmaxima_values[i] + potenciaEolicamaxima_values[i])

        # Calculando o custo total do dia
    Custo_dia = sum(CustoTotal) / 1000 / (288)
    status_bateria = status_bateria * 100 / pbat
    status_tanque = status_tanque * 100 / tanque_hidrogenio_max

    print("Otimização concluída, custo total.", Custo_dia)
    st.success(
        "Otimização concluída, custo total para implementação e operção durante um ano: R$ {:,.2f}".format(Custo_dia))
    st.markdown(f"Vida útil do sistema: {vida_util_sistema:.2f} anos")
    # Visualização com Plotly
    timesteps = range(1, len(PotenciaSolar) + 1)
    timesteps = list(timesteps)

    return potenciasolarmaxima_values, potenciaEolicamaxima_values, timesteps, PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal, PotenciaTotal, previsoes, status_bateria, status_tanque

def FazAsImagens(potenciasolarmaxima_values, potenciaEolicamaxima_values, timesteps, PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal, PotenciaTotal, previsoes, status_bateria, status_tanque):
    # Gráfico 1: Potências ao longo do tempo
    st.subheader("Potências ao longo do tempo")
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(x=timesteps, y=PotenciaSolar, mode='lines+markers', name='Potência Solar', line=dict(color='yellow'),
                   marker=dict(symbol='diamond', size=7, color='yellow')))
    fig1.add_trace(
        go.Scatter(x=timesteps, y=PotenciaEolica, mode='lines+markers', name='Potência Eólica', line=dict(color='blue'),
                   marker=dict(symbol='diamond', size=7, color='blue')))
    fig1.add_trace(
        go.Scatter(x=timesteps, y=PotenciaH2, mode='lines+markers', name='Potência H2', line=dict(color='green'),
                   marker=dict(symbol='diamond', size=7, color='green')))
    fig1.add_trace(go.Scatter(x=timesteps, y=PotenciaBateria, mode='lines+markers', name='Potência Bateria',
                              line=dict(color='purple'), marker=dict(color='purple')))
    fig1.update_layout(xaxis_title="Tempo em horas", yaxis_title="Potência (W)",
                       width=1500, height=500, legend=dict(x=0.95, y=1), template="plotly_white")
    st.plotly_chart(fig1)

    CriarGraficos(fig1, 1)

    # Gráfico 2: Custo Total
    st.subheader("Custo Total")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=timesteps, y=CustoTotal, mode='lines+markers', name='Custo Total',line=dict(color='black'),
                               marker=dict(symbol='diamond', size=7, color='black') ))
    #fig2.add_trace(go.Scatter(x=timesteps, y=potenciaEolicamaxima_values, mode='lines+markers', name='Vento',line=dict(color='blue'),
      #                        marker=dict(symbol='triangle-up', size=7, color='blue')))
    #fig2.add_trace(go.Scatter(x=timesteps, y=previsoes.flatten(), mode='lines+markers', name='Carga',
       #                       line=dict(color='red'), marker=dict(symbol='triangle-up', size=7, color='red')))
    fig2.update_layout(xaxis_title="Tempo em horas", yaxis_title="Custo R$")
    st.plotly_chart(fig2)
    CriarGraficos(fig2, 2)

    # Gráfico 3: Potência Total e Carga Residencial
    st.subheader("Potência das geraçoes e Carga ")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=timesteps, y=potenciasolarmaxima_values, mode='lines+markers', name='Solar',line=dict(color='#F9C61B'),
                               marker=dict(symbol='diamond', size=7, color='#F9C61B') ))
    fig3.add_trace(go.Scatter(x=timesteps, y=potenciaEolicamaxima_values, mode='lines+markers', name='Vento',line=dict(color='blue'),
                              marker=dict(symbol='triangle-up', size=7, color='blue')))
    fig3.add_trace(go.Scatter(x=timesteps, y=previsoes.flatten(), mode='lines+markers', name='Carga',
                              line=dict(color='red'), marker=dict(symbol='triangle-up', size=7, color='red')))
    fig3.update_layout(xaxis_title="Tempo em horas",
                       yaxis_title="Potência (W)",legend=dict(x=0.95, y=1),
                       width=1500, height=500)
    st.plotly_chart(fig3)

    CriarGraficos(fig3, 3)

    # Gráfico 4: Status da Bateria e Tanque
    st.subheader("Status da Bateria e Tanque")
    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(x=timesteps, y=status_bateria, mode='lines+markers', name='Bateria (%)', line=dict(color='purple'),
                   marker=dict(color='purple')))
    fig4.add_trace(go.Scatter(x=timesteps, y=status_tanque, mode='lines+markers', name='Tanque de H2 (%)',
                              line=dict(color='green'), marker=dict(color='green')))
    fig4.update_layout(xaxis_title="Tempo em horas",
                       yaxis_title="Status (%)",legend=dict(x=0.95, y=1),
                       width=1500, height=500)
    st.plotly_chart(fig4)

    CriarGraficos(fig4, 4)
    st.success("Previsão de carga e Despacho concluído!")

    


st.title("Previsão de Carga")

st.sidebar.subheader("Parâmetros da Rede Neural")
epochs = st.sidebar.number_input("Quantas epocas simular", value=50)
batch_size = st.sidebar.number_input("Tamanho do lote", value=32)
learning_rate = st.sidebar.number_input("Taxa de aprendizado", value=0.001, format="%.3f")
num_neurons = st.sidebar.number_input("Número de neurônios por camada", value=64)
activation = st.sidebar.selectbox("Função de ativação", ["relu", "sigmoid", "tanh"])
passos_anteriores = st.sidebar.number_input("Passos anteriores para previsão", value=10)

st.sidebar.subheader("Parâmetros do Despacho O&M")
preco_solar = st.sidebar.number_input("Preço Solar (U$/kW)", value=50)
preco_eolica = st.sidebar.number_input("Preço Eólica (U$/kW)", value=20)
preco_h2 = st.sidebar.number_input("Preço H2 (U$/kW)", value=20)
preco_bateria = st.sidebar.number_input("Preço Bateria (U$/kW)", value=10)
dolar_hoje = st.sidebar.number_input("Dólar Hoje", value=5.1)
Potenciadosistema = st.sidebar.number_input("Potência do Sistema (kW)", value=30000)
vida_util = st.sidebar.number_input("Vida útil do Sistema (anos)", value=20)



# Upload do arquivo de entrada
def carregar_dados():
    if "dados_carga" not in st.session_state:
        st.session_state["dados_carga"] = None

    uploaded_file = st.file_uploader("Carregue o arquivo de dados da carga (Excel)", type="xlsx")
    if uploaded_file:
        dados = pd.read_excel(uploaded_file)
        st.session_state["dados_carga"] = dados
        st.success("Arquivo carregado com sucesso!")
        if st.sidebar.button("Executar Previsão de Carga"):
            potenciasolarmaxima_values, potenciaEolicamaxima_values, timesteps, PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal, PotenciaTotal, previsoes, status_bateria, status_tanque = executar_codigo(dados, preco_solar, preco_eolica, preco_h2, preco_bateria, dolar_hoje, Potenciadosistema, epochs,
                            batch_size, learning_rate, num_neurons, activation, passos_anteriores)
            FazAsImagens(potenciasolarmaxima_values, potenciaEolicamaxima_values, timesteps, PotenciaSolar, PotenciaEolica, PotenciaH2, PotenciaBateria, CustoTotal, PotenciaTotal, previsoes, status_bateria, status_tanque)

    return st.session_state.get("dados_carga")

dados = carregar_dados()
if dados is None:
    st.warning("Por favor, carregue um arquivo Excel para continuar.")



image_url = "https://raw.githubusercontent.com/TiagoPitol/TCC/main/LogoLAPES1.png"

# Baixar a imagem do GitHub
response = requests.get(image_url)
encoded_string = base64.b64encode(response.content).decode()


# Exibir a imagem no canto inferior direito, sobre os demais elementos
st.markdown(
    f"""
    <div style="
        position: fixed;
        bottom: 10px;
        right: 10px;
        z-index: 1000;
    ">
        <img src="data:image/png;base64,{encoded_string}" width="200">
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <style>
        /* Fundo principal (modo Light padrão) */
        .stApp {
            background-color: #FFFFFF !important;
        }

        /* Texto padrão */
        h1, h2, h3, h4, h5, h6, p, label, div {
            color: #000000 !important;
        }

        /* Fundo da Sidebar */
        [data-testid="stSidebar"] {
            background-color: #9AD6CC !important;
        }

        /* Botões e elementos interativos */
        .stButton>button {
            background-color: #FFFFFF !important;
            color: #FFFFFF !important;
            border: 1px solid #CCCCCC !important;
        }

        .reportview-container {
            background-color: #fFfffF;
        }
        /* Alteração de links */
        a {
            color: #1E70BF !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
