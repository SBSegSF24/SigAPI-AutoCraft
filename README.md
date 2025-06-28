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

./demo_local.sh

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

python3 main.py --autocraft -d ../datasets/rus/androcrawl_rus.csv

```

- Run SigAPI **Original**
```bash

cd src

python3 main.py -d ../datasets/smote/drebin_215_smote.csv

```

## :test_tube: Test Environment

All experiments were validated on:

- **Hardware**: Intel Core i5-8265U, 8 core, 8 GB RAM. **Software**: Ubuntu 22.04.4 LTS, Kernel 6.5.0-35-generic, Python 3.10.12, Docker 24.0.5.  
- **Hardware**: Intel Core i7-10700, 8 cores, 16 GB RAM. **Software**: Ubuntu 24.02 LTS, Kernel 6.8.0-38-generic, Python 3.12.3, Docker 26.1.4.
- **Hardware**: AMD Ryzen 7 5800X 8-core, 64GB RAM. **Software**: Ubuntu Server 22.04.2 LTS, Kernel 6.2.0-33-generic, Python 3.10.12, Docker 24.07.

## :bar_chart: Datasets

P = Permissions, A = API Calls, I = Intents

| Dataset                | Features     | Baseline Benign | Baseline Malicious | SMOTE Per Class | RUS Per Class | Unique Benign | Unique Malicious |
|-------------------------------|--------------|-----------------|--------------------|-----------------|---------------|---------------|------------------|
| Adroit                 | 166 (P)      | 8,058           | 3,418              | 5,640           | 3,418         | 992           | 133              |
| AndroCrawl             | 81 (P, A)    | 86,562          | 10,170             | 60,593          | 10,170        | 13,850        | 3 249            |
| Android Permission     | 151 (P)      | 9,077           | 17,787             | 12,450          | 9,077         | 1,443         | 976              |
| DefenseDroid Closeness | 500 (A)      | 5,222           | 5,224              | 3,678           | 5,222         | 3,130         | 2,899            |
| DefenseDroid Degree    | 500 (A)      | 5,222           | 5,224              | 3,678           | 5,222         | 3,125         | 2,894            |
| DefenseDroid Katz      | 500 (A)      | 5,222           | 5,224              | 3,678           | 5,222         | 3,297         | 3,042            |
| DefenseDroid PRS        | 500 (P, I)   | 11,975          | 5,975              | 4,200           | 5,222         | 4,403         | 2,364            |
| Drebin-215              | 215 (P, A)   | 9,476           | 5,555              | 6,633           | 5,555         | 3,826         | 1,099            |
| KronoDroid Device       | 286 (P, A)   | 36,755          | 41,382             | 28,976          | 36,755        | 16,908        | 14,555           |
| MH-100K                 | 500 (P, I, A)| 89,213          | 12,721             | 62,493          | 12,721        | 23,468        | 7,437            |

## :link: References

#### [Original SigAPI Paper](https://galib19.github.io/publications/SigapiSEKE2020)
