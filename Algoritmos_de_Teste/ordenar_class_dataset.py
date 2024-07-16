import csv

# Abra o arquivo CSV para leitura
with open('dataset.csv', 'r') as arquivo_csv:
    leitor_csv = csv.reader(arquivo_csv)
    linhas = list(leitor_csv)

# Classifique as linhas com base no valor da última coluna (assumindo que são números)
linhas_ordenadas = sorted(linhas, key=lambda x: int(x[-1]))

# Abra o arquivo CSV para escrita
with open('dataset.csv', 'w', newline='') as arquivo_csv_ordenado:
    escritor_csv = csv.writer(arquivo_csv_ordenado)
    escritor_csv.writerows(linhas_ordenadas)

print("Arquivo CSV ordenado com sucesso.")