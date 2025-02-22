## SigAPI AutoCraft: um método otimizado e generalista de seleção de características para detecção de malware Android

Métodos de seleção de características são amplamente utilizados no contexto de detecção de malwares Android.
O SigAPI, semelhante a outros métodos de seleção de características, foi desenvolvido e avaliado utilizando apenas dois datasets e, consequentemente, apresentou problemas de generalização em diversos datasets no contexto de malwares Android.
O SigAPI apresenta dois desafios que dificultam sua aplicação prática: a necessidade de definir um número mínimo de características e a instabilidade das métricas. Além disso, não é eficiente em generalizar na diversidade de datasets Android.
Para resolver esses problemas, desenvolvemos uma versão aprimorada do método, denominada SigAPI AutoCraft, que consegue atingir resultados promissores em dez datasets de malware Android.
Os resultados indicam uma boa capacidade de generalização e um aumento no desempenho de predição em até 20%



## Preparação e instalação

### :octocat: Clonando o repositório Github
```bash

git clone https://github.com/SBSegSF24/SigAPI-AutoCraft.git

cd SigAPI-AutoCraft

```



### :whale: Instalação no ambiente Docker

1. Instalando o Docker:
```bash

sudo apt install docker docker.io

sudo usermod -aG docker $USER # necessário apenas se o usuário ainda não utilizar docker em modo usuário

```

2. Construíndo a imagem com o Docker:
```bash
docker build -t sigapiautocraft:latest .

```
### :penguin: Instalação no ambiente local

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

## :mouse: Execução demo

As demos executam o SigAPI AutoCraft em um único dataset reduzido ([dataset Adroit BL](https://github.com/SBSegSF24/SigAPI-AutoCraft/blob/cbeaf5872abe324db5510c361975999da86d044c/Datasets/Balanceados/adroit_bl.csv)). A demo leva **menos de 1 minuto** em uma máquina máquina _AMD Ryzen 7 5800X 8 cores com 64 GB de ram_.

**No ambiente docker**
~~~sh
./demo_docker.sh
~~~
**No ambiente local**
~~~sh
./demo_venv.sh
~~~

**Executando a ferramenta** (use **python 3.10.12** ou posterior)

## :dart: Reproduzindo os experimentos do trabalho

A execução das duas reproduções (SigAPI Original e SigAPI AutoCraft) pode levar até mais de 24 horas dependendo do hardware. 

### Execução do SigAPI Original para todos os datasets no ambiente local
 
```
./reproduzir_sigapi.sh
```

### Execução do SigAPI AutoCraft para todos os datasets no ambiente local
```
./reproduzir_sigapi_autocraft.sh
```
### Executando em um container em modo **persistente** ou **não persistente**

**Não persistente**: Os arquivos de saida serão apagados quando o conteiner finalizar a execução.

```bash

docker run -it sigapiautocraft

```
**Persistente**: Os arquivos de saída da execução serão salvos no diretório atual

```bash

docker run -v $(readlink -f .):/sigapi -it sigapiautocraft

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
python3 run_ml_RandomForest.py -d Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-adroit.csv -c rf
```

Para os demais datasets selecionados pelo método **original**, substitua o valor do parâmetro `-d` conforme a lista abaixo:
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-drebin_215_all.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-adroit.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasetss/ORIGINAL-resultado-selecao-balanceados-android_permissions.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-kronodroid_real_device.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-katz.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-degree.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-closeness.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-MH100K.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Original/Datasets/ORIGINAL-resultado-selecao-balanceados-defensedroid_prs.csv`

Para os demais datasets selecionados pelo método **otimizado**, substitua o valor do parâmetro `-d` conforme a lista abaixo:

- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/resultado-selecao-BALANCEADOS-adroit_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/resultado-selecao-BALANCEADOS-drebin_215_all_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/resultado-selecao-BALANCEADOS-android_permissions_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/resultado-selecao-BALANCEADOS-kronodroid_real_device_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/resultado-selecao-BALANCEADOS-katz_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/resultado-selecao-BALANCEADOS-degree_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/Datasets/resultado-selecao-BALANCEADOS-closeness_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/Datasets/resultado-selecao-BALANCEADOS-MH100K_bl.csv`
- `Resultados/Paper_SBSeg_Trilha_Principal/Otimizado/Datasets/Datasets/resultado-selecao-BALANCEADOS-defensedroid_prs_bl.csv`

***OBS: Recomenda-se armazenar os resultados gerados pelo método em uma pasta separada após cada execução.***


## :dash: Ambiente de testes

O método foi testado no seguinte ambiente:
- **Hardware**: Intel Core i5-8265U, 8 core, 8 GB RAM. **Software**: Ubuntu 22.04.4 LTS, Kernel 6.5.0-35-generic, Python 3.10.12, Docker 24.0.5.  
- **Hardware**: Intel Core i7-10700, 8 cores, 16 GB RAM. **Software**: Ubuntu 24.02 LTS, Kernel 6.8.0-38-generic, Python 3.12.3, Docker 26.1.4.
- **Hardware**: AMD Ryzen 7 5800X 8-core, 64GB RAM. **Software**: Ubuntu Server 22.04.2 LTS, Kernel 6.2.0-33-generic, Python 3.10.12, Docker 24.07.

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

## Datasets:
|             Dataset             | Nº Amostras  | Nº Amostras balanceado | Nº Features  | Nº Features balanceado |
|:-------------------------------:|:------------:|:----------------------:|:------------:|:----------------------:|
|            Androcrawl           |    162983    |          20340         |      221     |           82           |
|              Adroit             |     11476    |          6836          |      182     |           167          |
|            Drebin-215           |     15031    |          11110         |      215     |           209          |
|       Android Permissions       |     29999    |          18154         |      183     |           152          |
|      Kronodroid Real Device     |     78137    |          73510         |      483     |           287          |
|              MH100K             |    101,975   |          20000         |     24833    |           201          |
| Defensedroid Apicalls Closeness |     10476    |          10444         |     21997    |           201          |
|   Defensedroid Apicalls Degree  |     10476    |          10444         |     21997    |           201          |
|    Defensedroid Apicalls Katz   |     10476    |          10444         |     21997    |           201          |
|         Defensedroid prs        |     11975    |          11950         |     2938     |           201          |


## Outras informações

#### [Link para o artigo do método SigAPI Original](https://galib19.github.io/publications/SigapiSEKE2020)
