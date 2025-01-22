import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
import streamlit as st

# Configurações do Streamlit
st.title("Análise de Otimização de Energia")
st.subheader("Simulação de uma Microrrede com geração solar, eólica, H2 e bateria")

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Carregue o arquivo 'DadosDeEntrada.xlsx'", type="xlsx")
if uploaded_file:
    # Leitura dos dados
    geracao = pd.read_excel(uploaded_file).values

    # Parâmetros iniciais
    G_Solar = np.zeros(288)
    Radiacao = np.array(geracao[18:306, 7])
    Temperatura = np.array(geracao[18:306, 6])
    Vento = np.array(geracao[18:306, 4])

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
        P_WT[i] = n_WT * eta_WT * P_R_WT * ((u**2 - u_cut_in**2) / (u_rated**2 - u_cut_in**2))
    elif u_rated < u < u_cut_off:
        P_WT[i] = n_WT * eta_WT * P_R_WT
    else:
        P_WT[i] = 0

# Parâmetros gerais
potenciadosistema = 25000
potenciaEolicamaxima_values = P_WT * potenciadosistema
potenciasolarmaxima_values = G_Solar * potenciadosistema
Carga_Residencial = geracao[18:306, 12] * 30000

precoH2 = 14.3
precoeolica = 0.1712
precosolar = 0.17566
precobateria = 0.01
tanque_hidrogenio_max = 20
nivel_tanque = tanque_hidrogenio_max
consumo_kg_h2 = 0.2
eficiencia_eletrolisador = 0.75
consumo_kW_h2 = -53571*eficiencia_eletrolisador + 92643
potencia_maxima_eletrolisador = 40000
niveis_eletrolisador = 6

# Possibilidades do eletrólise
for n in range(1, niveis_eletrolisador + 1):
    if n==1:
        possibilidades_eletrolisador = [0]
    else:
        possibilidades_eletrolisador = [0] + [i * potencia_maxima_eletrolisador / niveis_eletrolisador for i in range(1, niveis_eletrolisador + 1)]


# Inicialização
pbat = 30000
potenciaBateria_max = pbat
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
    
    if potenciaBateria_max > 3000:
        # Problema de otimização
        c = [precosolar, precoeolica , precoH2, precobateria ]
        A_eq = [[1, 1, 1, 1]]
        b_eq = [Carga_Residencial[i]]
        bounds = [
            (0, potenciasolarmaxima),
            (0, potenciaEolicamaxima),
            (0, potenciaH2maxima),
            (potenciaBateria_min, potenciaBateria_max),
        ]
        # Resolvendo o problema de otimização
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
    else:
        # Problema de otimização
        c = [precosolar, precoeolica , precoH2, 0 ]
        A_eq = [[1, 1, 1,1]]
        b_eq = [Carga_Residencial[i]]
        bounds = [
            (0, potenciasolarmaxima),
            (0, potenciaEolicamaxima),
            (0, potenciaH2maxima),
            (0,0),
                ]
        # Resolvendo o problema d
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    PotenciaBateria.append(result.x[3])
    PotenciaSolar.append(result.x[0])  
    PotenciaEolica.append(result.x[1])  
    CustoTotal.append(result.fun)  # Adicionando o valor da função objetivo
    PotenciaH2.append(result.x[2])  
       

    possivel_carga_bateria = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i]
    possivel_carga_H2 = potenciasolarmaxima + potenciaEolicamaxima - Carga_Residencial[i] - possivel_carga_bateria
    if possivel_carga_bateria < 0:
        possivel_carga_bateria = 0
    if possivel_carga_H2 < 0:
        possivel_carga_H2 = 0
    if possivel_carga_bateria > pbat - potenciaBateria_max:
        possivel_carga_bateria = pbat - potenciaBateria_max #igual ao valor da diferenca entre a capacidade maxima e a capacidade atual

    if nivel_tanque >= tanque_hidrogenio_max:
        possivel_carga_H2 = 0
    else:
        for n in range(2, niveis_eletrolisador + 2):
            if possivel_carga_H2 < possibilidades_eletrolisador[n - 1]:
                possivel_carga_H2 = possibilidades_eletrolisador[n - 2]
            if n == (niveis_eletrolisador + 1) and possivel_carga_H2 >= possibilidades_eletrolisador[n - 1]:
                possivel_carga_H2 = possibilidades_eletrolisador[n - 1]
    if tanque_hidrogenio_max - nivel_tanque < possivel_carga_H2/consumo_kW_h2:
        possivel_carga_H2 = ((tanque_hidrogenio_max - nivel_tanque)*consumo_kW_h2)

       
    if result.success:
        x = result.x
        potenciaBateria_max = potenciaBateria_max - x[3] + possivel_carga_bateria
        #potenciaBateria_max1 = max(potenciaBateria_max, 3000)
        nivel_tanque = (nivel_tanque - (x[2] / 1000) * consumo_kg_h2 + (possivel_carga_H2 /consumo_kW_h2))
    if potenciaBateria_max < 3000:
        potenciaBateria_max = 3000 
        
PotenciaTotal = []           
for i in range(len(PotenciaEolica)):
    PotenciaTotal.append(potenciasolarmaxima_values[i] + potenciaEolicamaxima_values[i])

          # Calculando o custo total do dia
Custo_dia = sum(CustoTotal)/1000
status_bateria = status_bateria*100/pbat
status_tanque = status_tanque*100/tanque_hidrogenio_max
print("Otimização concluída, custo total.", Custo_dia)

    # Visualização com Plotly
timesteps = range(1, len(PotenciaSolar) + 1)
timesteps = list(timesteps)
    # Gráfico 1: Potências ao longo do tempo
# Função para salvar e criar botão de download
def save_and_create_button(fig, graph_num):
    # Salvando as imagens nos diferentes formatos
    output_file_jpeg = f"grafico_{graph_num}_high_quality.jpg"
    output_file_png = f"grafico_{graph_num}_high_quality.png"
    output_file_svg = f"grafico_{graph_num}_high_quality.svg"

    # Salvando a imagem em diferentes formatos
    fig.write_image(output_file_jpeg, format='jpeg', scale=3)
    fig.write_image(output_file_png, format='png', scale=3)
    fig.write_image(output_file_svg, format='svg', scale=3)

    # Selecionando o formato para download
    format_option = st.selectbox(
        f"Escolha o formato para baixar o gráfico {graph_num}",
        ['Selecione', 'JPEG', 'PNG', 'SVG']
    )

    # Criando o botão de download baseado na escolha do formato
    if format_option == 'JPEG':
        with open(output_file_jpeg, "rb") as file:
            st.download_button(
                label=f"Baixar Gráfico {graph_num} em JPEG",
                data=file,
                file_name=f"grafico_{graph_num}_high_quality.jpg",
                mime="image/jpeg"
            )
    elif format_option == 'PNG':
        with open(output_file_png, "rb") as file:
            st.download_button(
                label=f"Baixar Gráfico {graph_num} em PNG",
                data=file,
                file_name=f"grafico_{graph_num}_high_quality.png",
                mime="image/png"
            )
    elif format_option == 'SVG':
        with open(output_file_svg, "rb") as file:
            st.download_button(
                label=f"Baixar Gráfico {graph_num} em SVG",
                data=file,
                file_name=f"grafico_{graph_num}_high_quality.svg",
                mime="image/svg+xml"
            )

 
# Gráfico 1
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=timesteps, y=PotenciaSolar, mode='lines+markers', name='Potência Solar', line=dict(color='yellow'), marker=dict(color='yellow')))
fig1.add_trace(go.Scatter(x=timesteps, y=PotenciaEolica, mode='lines+markers', name='Potência Eólica', line=dict(color='blue'), marker=dict(color='blue')))
fig1.add_trace(go.Scatter(x=timesteps, y=PotenciaH2, mode='lines+markers', name='Potência H2', line=dict(color='green'), marker=dict(color='green')))
fig1.add_trace(go.Scatter(x=timesteps, y=PotenciaBateria, mode='lines+markers', name='Potência Bateria', line=dict(color='purple'), marker=dict(color='purple')))
fig1.update_layout(title="Potências ao Longo do Tempo", xaxis_title="Etapas de Tempo", yaxis_title="Potência (kW)", width=1200, height=450, template="plotly_white")
st.plotly_chart(fig1)

save_and_create_button(fig1, 1)

# Gráfico 2: Custo Total
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=timesteps, y=CustoTotal, mode='lines+markers', name='Custo Total'))
fig2.update_layout(title="Custo Total ao Longo do Tempo", xaxis_title="Etapas de Tempo", yaxis_title="Custo ($)")
st.plotly_chart(fig2)

save_and_create_button(fig2, 2)

# Gráfico 3: Potência Total e Carga Residencial
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=timesteps, y=PotenciaTotal, mode='lines+markers', name='Potência Total', line=dict(color='blue')))
fig3.add_trace(go.Scatter(x=timesteps, y=Carga_Residencial, mode='lines+markers', name='Carga Residencial', line=dict(color='green')))
fig3.update_layout(title="Potência Total vs Carga Residencial", xaxis_title="Etapas de Tempo", yaxis_title="Potência (kW)")
st.plotly_chart(fig3)

save_and_create_button(fig3, 3)

# Gráfico 4: Status da Bateria e Tanque
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=timesteps, y=status_bateria, mode='lines+markers', name='Bateria (%)'))
fig4.add_trace(go.Scatter(x=timesteps, y=status_tanque, mode='lines+markers', name='Tanque de H2 (%)'))
fig4.update_layout(title="Status da Bateria e Tanque de H2", xaxis_title="Etapas de Tempo", yaxis_title="Status (%)")
st.plotly_chart(fig4)

# Salvando e criando os botões para cada gráfico
save_and_create_button(fig4, 4)

st.success("Simulação concluída!")
