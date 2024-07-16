## SigAPI AutoCraft: um método otimizado e generalista de seleção de características para detecção de malware Android

Métodos de seleção de características são amplamente utilizados no contexto de detecção de malwares Android.
O SigAPI, semelhante a outros métodos de seleção de características, foi desenvolvido e avaliado utilizando apenas dois datasets e, consequentemente, apresentou problemas de generalização em diversos datasets no contexto de malwares Android.
O SigAPI apresenta dois desafios que dificultam sua aplicação prática: a necessidade de definir um número mínimo de características e a instabilidade das métricas. Além disso, não é eficiente em generalizar na diversidade de datasets Android.
Para resolver esses problemas, desenvolvemos uma versão aprimorada do método, denominada SigAPI AutoCraft, que consegue atingir resultados promissores em dez datasets de malware Android.
Os resultados indicam uma boa capacidade de generalização e um aumento no desempenho de predição em até 20%

## [Artigo SigAPI Original](https://galib19.github.io/publications/SigapiSEKE2020)

## Clonando o repositório Github
```bash

git clone https://github.com/SBSegSF24/SigAPI-AutoCraft.git

cd SF24_SigAPI

```

## :whale: Executando em Docker

1. Instalando o Docker:
```bash

sudo apt install docker docker.io

sudo usermod -aG docker $USER # necessário apenas se o usuário ainda não utilizar docker em modo usuário

```

2. Construíndo a imagem com o Docker:
```bash
docker build -t sigapi:latest .

```

3. Iniciando um container em modo **persistente** ou **não persistente**

**Não persistente**: Os arquivos de saida serão apagados quando o conteiner finalizar a execução.

```bash

docker run -it sigapi

```
**Persistente**: Os arquivos de saída da execução serão salvos no diretório atual

```bash

docker run -v $(readlink -f .):/sigapi -it sigapi

```

## :penguin: Executando no seu Linux

**Instalando Python, se necessário**

~~~sh
sudo apt update

sudo apt install python3
~~~

**Instalando o gerenciado padrão de pacotes do Python (pip), se necessário**

~~~sh
sudo apt install python3-pip
~~~

**Use ambiente virtual Python**

~~~sh
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
~~~

**Instalando pacotes necessários**

~~~sh
pip install -r requirements.txt
~~~

**Executando a ferramenta** (use **python 3.10.12** ou posterior)


## :dart: Reprodução dos experimentos

### Execução do SigAPI Original para todos os datasets

```
./reproduzir_sigapi.sh
```

### Execução do SigAPI AutoCraft para todos os datasets
```
./reproduzir_sigapi_autocraft.sh
```

## :shell: Execução manual do SigAPI AutoCraft para um dataset

### Exemplo:
Parâmetros:
`-d` indica dataset
`-o` indica output
`-i` indica incremento
```
python3 -m SigAPI_Otimizado.metodos.SigAPI.main -d Datasets/Balanceados/androcrawl_all_bl.csv -o resultado-selecao-BALANCEADOS-androcrawl_all_bl.csv -i 1
```

Para os demais datasets, substitua o valor do parâmetro `-d` conforme a lista abaixo:
- `Datasets/Balanceados/adroit_bl.csv`
- `Datasets/Balanceados/drebin_215_all_bl.csv`
- `Datasets/Balanceados/android_permissions_bl.csv`
- `Datasets/Balanceados/kronodroid_real_device_bl.csv`
- `Datasets/Balanceados/reduced_balanced_defensedroid_apicalls_closeness.csv`
- `Datasets/Balanceados/reduced_balanced_defensedroid_apicalls_degree.csv`
- `Datasets/Balanceados/reduced_balanced_defensedroid_apicalls_katz.csv`
- `Datasets/Balanceados/reduced_20k_mh_100k_filtered.csv`
- `Datasets/Balanceados/reduced_balanced_defensedroid_prs.csv`

## :shell: Execução manual do Random Forest para um dataset reduzido

### Exemplo:
```
python3 run_ml_RandomForest.py -d Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-adroit.csv -c rf
```

Para os demais datasets selecionados pelo método **original**, substitua o valor do parâmetro `-d` conforme a lista abaixo:
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-drebin_215_all.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-adroit.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-android_permissions.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-kronodroid_real_device.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-katz.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-degree.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-closeness.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-MH100K.csv`
- `Resultados/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-defensedroid_prs.csv`

Para os demais datasets selecionados pelo método **otimizado**, substitua o valor do parâmetro `-d` conforme a lista abaixo:

- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-adroit_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-drebin_215_all_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-android_permissions_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-kronodroid_real_device_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-katz_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-degree_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-closeness_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-MH100K_bl.csv`
- `Resultados/Otimizado/Datasets/resultado-selecao-BALANCEADOS-defensedroid_prs_bl.csv`

***OBS: Recomenda-se armazenar os resultados gerados pelo método em uma pasta separada após cada execução.***


# Ambiente de testes

O método foi testado no seguinte ambiente:
- Notebook Intel Core i5-8265U da 8ª geração, CPU @1.60GHz (8 núcleos) e 8GB de memória RAM
- Sistema operacional Ubuntu 22.04.4 LTS com Kernel 6.5.0-35-generic


O repositório está organizado da seguinte maneira:

```
/Algoritmos_de_Teste
/Datasets
/Documentos
/Modelos_gerados
/Resultados
    /Datasets_Balanceados_Reduzidos - Desconsiderar
    /Metricas_50_Features
    /Metricas_100_Features
    /Original
    /Otimizado
    /PCA_vs_InfoGain
/SigAPI_Original
/SigAPI_Otimizado

- README.md
```

## Descrição dos Diretórios:

- **Algoritmos_de_Teste**: Scripts relacionados aos algoritmos e testes.

- **Datasets**: Contém todos os datasets utilizados.

- **Modelos_gerados**: Contém os modelos treinados salvos.

- **Resultados**: Armazena resultados dos testes e execuções.
  - **Metricas_50_Features**: Diretório para armazenar métricas relacionadas a 50 features.
  - **Metricas_100_Features**: Diretório para armazenar métricas relacionadas a 100 features.
  - **Original**: Contém resultados da versão original.
  - **Otimizado**: Contém resultados da versão otimizada.
  - **PCA_vs_InfoGain**: Diretório para análises comparativas entre PCA e InfoGain.

- **SigAPI_Original**: Contém arquivos e scripts relacionados à versão original do SigAPI.

- **SigAPI_Otimizado**: Contém arquivos e scripts relacionados à versão otimizada do SigAPI.

### Adicionar uma tabela do balanceamento ex:
| Dataset                | Benignos | Malwares | Nº Features |
|------------------------|----------|----------|-------------|
| Androcrawl             | 1500     | 1500     | 300         |
| Adroit                 | 2000     | 2000     | 250         |
| Drebin-215             | 1000     | 1000     | 400         |
| Android Permissions    | 1200     | 1200     | 350         |

### Seria interessante colocar uma tabela comparativa simples como no exemplo abaixo:

| Dataset | Método     | Nº Features | Acurácia | Recall | Modelo |
|---------|------------|-------------|----------|--------|--------|
| adroit  | Original   | 37          | 94.2%    | 93.5%  | RF     |
| adroit  | Modificado | 22          | 95.2%    | 97.5%  | RF     |
| drebin  | Original   | 45          | 91.5%    | 92.8%  | SVM    |
| drebin  | Modificado | 30          | 93.8%    | 95.1%  | SVM    |
| android_permissions | Original   | 55          | 88.6%    | 87.3%  | MLP    |
| android_permissions | Modificado | 40          | 90.1%    | 91.5%  | MLP    |
