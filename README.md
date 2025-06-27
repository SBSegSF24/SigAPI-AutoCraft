## Generalizing Feature Selection in Android Malware Detection: The SigAPI AutoCraft Approach

Feature selection methods are widely used in Android malware detection to enhance accuracy and efficiency by isolating the most relevant features. However, their generalizability is often limited---approaches such as SigAPI are typically developed and evaluated on only a small number of datasets, limiting their performance across diverse scenarios. The practical application of SigAPI is further constrained by the need to predefine a minimum number of features, the instability of its evaluation metrics, and its inefficiency in adapting to the heterogeneity commonly found in Android datasets.
To mitigate these limitations, we have developed SigAPI AutoCraft, an enhanced and automated version of the method. SigAPI AutoCraft demonstrates promising results across ten Android malware datasets, significantly improving generalization. Our findings underscore its robust generalization capabilities and up to a 20% improvement in prediction performance.


## :hammer_and_wrench: Setup & Installation

### :octocat: Clone the GitHub Repository
```bash

git clone https://github.com/SBSegSF24/SigAPI-AutoCraft.git

cd SigAPI-AutoCraft

```

### :whale: Docker Environment

1. Install Docker:
```bash

sudo apt update

sudo apt install docker docker.io

sudo usermod -aG docker $USER

```

2. Build the Docker image:
```bash
docker build -t sigapiautocraft:latest .

```
### :penguin: Local Environment

1. Install Python 3 (if necessary)
```bash

sudo apt update

sudo apt install -y python3 python3-venv python3-pip

```

2. Create and activate a virtual environment
```bash

python3 -m venv venv

source venv/bin/activate

```

3. Install required Python packages
```bash

pip install -r requirements.txt

```

## :desktop_computer: Demo Execution

We provide a quick demo on a small, balanced dataset ([Demo Dataset](https://github.com/SBSegSF24/SigAPI-AutoCraft/blob/main/datasets/demo.csv)). On an AMD Ryzen 7 5800X (8 cores, 64 GB RAM), the demo completes in under 1 minute.

- **Using Docker**
```bash

./demo_docker.sh

```

- **Locally (virtual environment)**
```bash

./demo_venv.sh

```

## :pushpin: Available arguments:

```
usage: main.py [-h] -d DATASET [-c CLASS_COLUMN] [--parallelize] [--output OUTPUT] [-th THRESHOLD] [-ifp INITIAL_FEATURES_PERCENT] [-pi PERCENT_INCREMENT] [--autocraft] [-m METRIC]

Optional Arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset (csv Files)
  -c CLASS_COLUMN, --class-column CLASS_COLUMN
                        Name of Class Column. Default: class
  --parallelize         Parallel Execution
  --output OUTPUT       Output File Directory. Default: results
  -th THRESHOLD, --threshold THRESHOLD
                        Threshold of Difference Between Metrics at Each Increment in Number of Features. When All Metrics Are Less Than It, Selection Phase Finishes. Default: 0.03
  -ifp INITIAL_FEATURES_PERCENT, --initial-features-percent INITIAL_FEATURES_PERCENT
                        Initial Features Percentage. Default: 0.05
  -pi PERCENT_INCREMENT, --percent-increment PERCENT_INCREMENT
                        Percentage to Increment Number of Features. Default: 0.05
  --autocraft           Run SigAPI Autocraft
  -m METRIC, --metric METRIC
                        Metric to Compare. Default: median. Choices: ['median', 'area', 'distance']
```


## :shell: Manual Usage for a Single Dataset

- Run SigAPI **AutoCraft**
```bash

cd src

python3 main.py --autocraft -d ../datasets/androcrawl.csv -o results -pi 0.1

```

- Run SigAPI **Original**
```bash

cd src

python3 main.py -d ../datasets/androcrawl.csv -o results -pi 0.1

```

- Run Random Forest Classifier
```bash

python3 run_ml.py --classifier rf -d results/autocraft_androcrawl.csv

```

## :test_tube: Test Environment

All experiments were validated on:

- **Hardware**: Intel Core i5-8265U, 8 core, 8 GB RAM. **Software**: Ubuntu 22.04.4 LTS, Kernel 6.5.0-35-generic, Python 3.10.12, Docker 24.0.5.  
- **Hardware**: Intel Core i7-10700, 8 cores, 16 GB RAM. **Software**: Ubuntu 24.02 LTS, Kernel 6.8.0-38-generic, Python 3.12.3, Docker 26.1.4.
- **Hardware**: AMD Ryzen 7 5800X 8-core, 64GB RAM. **Software**: Ubuntu Server 22.04.2 LTS, Kernel 6.2.0-33-generic, Python 3.10.12, Docker 24.07.

## :file_folder: Repository Structure

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

**Directory Descriptions:**

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

## :bar_chart: Datasets

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


## :link: References

#### [Original SigAPI Paper](https://galib19.github.io/publications/SigapiSEKE2020)
