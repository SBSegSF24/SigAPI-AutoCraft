# SigAPI

Neste espaço foi realizada a reprodução do artigo [Significant API Calls in Android Malware Detection](https://ksiresearch.org/seke/seke20paper/paper143.pdf).
Este artigo utiliza técnicas de seleção de características e a correlação baseada na eliminação destas features.

# Descrição
1 - Para selecionar as features são utilizadas 5 funções de seleção. Dentre estas cinco, a que possui o menor intervalo, neste caso, apresenta maior eficiência é a escolhida.

2 - Depois de encontrar a função mais eficiente para determinado conjunto de dados, é feita a correlação no intervalo indicado, para haver uma maior redução destas características.

## Como instalar

```
## Instale as seguintes dependências:
pip install pandas numpy scikit-learn
```

## Como rodar

Mude para o diretório raiz deste repositório (i.e.: `cd feature_selection`).

Primeiro iremos rodar as funções de selecao `sigapi_funcoesdeselecao.py`. 
Para rodar esta parte do experimento sobre algum dataset (e.g. `data.csv`) execute o seguinte comando:

```
python3 -m metodos.SigAPI.sigapi_funcoesdeselecao -d data.csv
``` 
         
Com isso, vão ser obtidos gráficos e dataframes sobre cada uma das 6 técnicas de seleção utilizadas.
Ao analisar esses dados, é possivel obter a técnica mais eficiente e seu intervalo de redução.

Após esta parte, é necessário rodar o código `sigapi_correlação.py` , fazendo o seguinte:

```
python3 -m metodos.SigAPI.sigapi_correlacao -d data.csv -k num_features -m method
``` 
onde esse k receberia o número de features para qual o dataset poderia ser reduzido e m seria o método que foi mais eficiente na redução.           
Com isso, vai ser possível encontrar a redução de características que foi realizada e o resultado será um dataset com estas características.
  
## Detalhes de uso
### Etapa de seleção de características
```
usage: sigapi_funcoesdeselecao.py [-h] -d DATASET [--sep SEPARATOR] [-c CLASS_COLUMN] [-n N_SAMPLES] [-o OUTPUT_FILE] [-t THRESHOLD] [-f INITIAL_N_FEATURES] [-i INCREMENT]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset (csv file). It should be already preprocessed.
  --sep SEPARATOR       Dataset feature separator. Default: ","
  -c CLASS_COLUMN, --class-column CLASS_COLUMN
                        Name of the class column. Default: "class"
  -n N_SAMPLES, --n-samples N_SAMPLES
                        Use a subset of n samples from the dataset. By default, all samples are used.
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file name. Default: results.csv
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold for the difference between metrics at each increment on the number of features. When all metrics are less than it, the selection phase finishes.
                        Default: 0.03
  -f INITIAL_N_FEATURES, --initial-n-features INITIAL_N_FEATURES
                        Initial number of features. Default: 1
  -i INCREMENT, --increment INCREMENT
                        Value to increment the initial number of features. Default: 1
```
### Etapa de correlação

```
usage: sigapi_correlacao.py [-h] -d DATASET [--sep SEPARATOR] [-c CLASS_COLUMN] [-n N_SAMPLES] -k NUM_FEATURES -m METHOD

options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset (csv file). It should be already preprocessed, with the last feature being the class
  --sep SEPARATOR       Dataset feature separator. Default: ","
  -c CLASS_COLUMN, --class-column CLASS_COLUMN
                        Name of the class column. Default: "class"
  -n N_SAMPLES, --n-samples N_SAMPLES
                        Use a subset of n samples from the dataset. RFG uses the whole dataset by default.
  -k NUM_FEATURES, --num_features NUM_FEATURES
                        Number of features
  -m METHOD, --method METHOD
                        One of the following feature selection methods to use: MutualInformationGain, RandomForestClassifier, ExtraTreesClassifier, RFERandomForestClassifier,
                        RFEGradientBoostingClassifier, SelectKBest
```
