## Artefatos apêndice SBSeg24/SF:  #243361: SigAPI AutoCraft: uma ferramenta de seleção de características com capacidade de generalização 

Métodos de seleção de características são amplamente utilizados no contexto de detecção de malwares Android.
O SigAPI, semelhante a outros métodos de seleção de características, foi desenvolvido e avaliado utilizando apenas dois datasets e, consequentemente, apresentou problemas de generalização em diversos datasets no contexto de malwares Android.
O SigAPI apresenta dois desafios que dificultam sua aplicação prática: a necessidade de definir um número mínimo de características e a instabilidade das métricas. Além disso, não é eficiente em generalizar na diversidade de datasets Android.
Para resolver esses problemas, desenvolvemos uma versão aprimorada do método, denominada SigAPI AutoCraft, que consegue atingir resultados promissores em dez datasets de malware Android.
Os resultados indicam uma boa capacidade de generalização e um aumento no desempenho de predição em até 20%


## 1. Selos Considerados

Os autores julgam como considerados no processo de avaliação os seguintes selos:


**SeloD - Artefatos Disponíveis**: Repositório GitHub público com documentação do artefato.

**SeloF - Artefatos Funcionais**: Artefato funcional e testado em Ubuntu 22.04 (bare metal).

**SeloR - Artefatos Reprodutíveis**: Scripts disponíveis para reprodução dos experimentos detalhados no artigo.

**SeloS- Artefatos Sustentáveis**: Código inteligível e acompanhado com boa documentação.

## 2. Informações básicas

Os códigos utilizados para a execução ferramenta, estão disponibilizados no repositório GitHub [https://github.com/SBSegSF24/SigAPI-AutoCraft](https://github.com/SBSegSF24/SigAPI-AutoCraft). No repositório há um **README.md** contendo informações sobre a  execução da ferramenta, configuração, parâmetros de entrada e instalação. O artefato foi testado em Ubuntu 22.04 com Python 3.10.12.

### 2.1. Dependências

O código da SigAPI AutoCraft possui dependências com diversos pacotes e bibliotecas Python.

Entre elas, as principais são:

- scikit-learn: 1.4.0

- pandas: 2.2.0

- matplotlib: 3.5.1

- numpy: 1.26.3

- joblib: 1.3.2

- imblearn: 0.0

A a lista completa das dependências encontra-se no arquivo **requirements.txt** do GitHub.

## 3. Instalação

Para instalar a ferramenta SigAPI AutoCraft basta clonar o repositório GitHub e seguir o passo-a-passo disponível no **README.md**.


## 4. Datasets

O diretório **Datasets** contém os conjuntos de dados originais utilizados no artigo. O diretório também contém os datasets balancedados e reduzidos.


## 5. Ambiente de testes

A ferramenta foi testada em Notebook Intel Core i5-8265U da 8ª geração, CPU @1.60GHz (8 núcleos) e 8GB de memória RAM, rodando Ubuntu Server 22.04 e Python 3.10.12.


## 6. Teste mínimo

A execução do teste mínimo funcional está documentada no **README.md**.

## 7. Experimentos

A reprodução pode ser realizada através de script e instruções no **README.md**.

