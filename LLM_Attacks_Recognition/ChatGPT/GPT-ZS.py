import pandas as pd
import time
import requests

#---------------------------------------------------------------------------------------------------------------------
# Configuração da API OpenAI
KEY = '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'
URL = "https://api.openai.com/v1/chat/completions"  # Atualizado para o endpoint correto
HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json"
}

SEMENTE = 42
#---------------------------------------------------------------------------------------------------------------------
# Leitura dos conjuntos de dados
data_attack_path = '/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SWaT_Dataset_Attack_v0.csv'
data_attack = pd.read_csv(data_attack_path)

#---------------------------------------------------------------------------------------------------------------------
# Read context
context_path = "/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SwatDataDescription.txt" 
paper_path = "/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SWaT_Paper.txt" 
context = ""
paper = ""
with open(context_path, "r") as file:
    context = file.read()

with open(paper_path, "r") as file:
    paper = file.read()

#---------------------------------------------------------------------------------------------------------------------
# Seleção de exemplos few-shot
sample_size = 10
exemplos_attacks = data_attack[data_attack['Normal/Attack'] == 'Attack'].sample(sample_size, random_state=SEMENTE)
exemplos_normais = data_attack[data_attack['Normal/Attack'] == 'Normal'].sample(sample_size, random_state=SEMENTE)

#---------------------------------------------------------------------------------------------------------------------
dados_teste = data_attack.sample(sample_size*2, random_state=SEMENTE)
y = dados_teste.iloc[:, -1]
dados_teste = dados_teste.drop(labels='Normal/Attack', axis=1)
     
#---------------------------------------------------------------------------------------------------------------------
# Criação das mensagens iniciais
instancias = [{"role": "system", "content": "The following texts contains the physical properties related to a water plant and the water treatment process, as well as network traffic in the testbed. The data of both physical properties and network traffic contains attacks in Cyber Physical Systems. Identify the texts with 'Attack' or 'Normal'"}]
instancias.append({"role": "system", "content": f"This is the context: {paper}"})

for row in exemplos_attacks.itertuples(index=False):
    data_tuple = tuple(row)
    attributes = [f"{column}: {value}" for column, value in zip(exemplos_attacks.columns, data_tuple)]
    data_string = " | ".join(attributes) +'\n'
    instancias.append({"role": "user", "content": f"The following text contains the physical properties related to a water plant and the water treatment process, as well as network traffic in the testbed. The data of both physical properties and network traffic contains attacks in Cyber Physical Systems. Identify the text with 'Attack' or 'Normal': {data_string}"})
    instancias.append({"role": "assistant", "content": "Attack"})

for row in exemplos_normais.itertuples(index=False):
    data_tuple = tuple(row)
    attributes = [f"{column}: {value}" for column, value in zip(exemplos_normais.columns, data_tuple)]
    data_string = " | ".join(attributes) +'\n'
    instancias.append({"role": "user", "content": f"The following text contains the physical properties related to a water plant and the water treatment process, as well as network traffic in the testbed. The data of both physical properties and network traffic contains attacks in Cyber Physical Systems. Identify the text with 'Attack' or 'Normal': {data_string}"})
    instancias.append({"role": "assistant", "content": "Normal"})
    
#---------------------------------------------------------------------------------------------------------------------
# Função para classificar texto com few-shot e Retry com Exponential Backoff
def classificar_texto(texto, max_retries=5):
    chat_messages = instancias + [{"role": "user", "content": f"The following text contains the physical properties related to a water plant and the water treatment process, as well as network traffic in the testbed. The data of both physical properties and network traffic contains attacks in Cyber Physical Systems. Identify the text with 'Attack' or 'Normal': {texto}"}]

    data = {
        "model": "gpt-4-1106-preview",
        "messages": chat_messages
    }
    
    for i in range(max_retries):
        resposta = requests.post(URL, headers=HEADERS, json=data)
        if resposta.status_code == 200:
            return resposta.json()['choices'][0]['message']['content'].strip()
        elif resposta.status_code == 429:  # Rate limited
            sleep_time = 2 ** (i + 1)  # Exponential backoff: 2, 4, 8, 16, ...
            print(f"Rate limited! Tentando novamente em {sleep_time} segundos...")
            time.sleep(sleep_time)
        else:
            print(f"Erro na requisição: {resposta.status_code} - {resposta.text}")
            sleep_time = 2 ** (i + 1)
            print(f"Esperando {sleep_time} segundos antes da próxima tentativa...")
            time.sleep(sleep_time)

    raise Exception("Número máximo de tentativas alcançado!")

#---------------------------------------------------------------------------------------------------------------------
tuples = []
for row in dados_teste.itertuples(index=False):
    data_tuple = tuple(row)
    attributes = [f"{column}: {value}" for column, value in zip(dados_teste.columns, data_tuple)]
    data_string = " | ".join(attributes) +'\n'
    tuples.append(data_string)

#---------------------------------------------------------------------------------------------------------------------
# Classificar cada tweet usando few-shot
resultados = []
classification = []
for i in tuples:
    resultado = classificar_texto(i)
    print(i)
    print(resultado + '\n')
    '''try:
        classification.append(resultado['choices'])
    finally:'''
    resultados.append(resultado)
    

#---------------------------------------------------------------------------------------------------------------------
# Salvar resultados
dados_teste['predictions'] = resultados
'''dados_teste['class'] = classification'''
dados_teste.to_csv('/media/work/icarovasconcelos/LLM_Attacks_Recognition/ChatGPT/ZeroShotResults.csv', index=False)
print("Classificação concluída e resultados salvos.")