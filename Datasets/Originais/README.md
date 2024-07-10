# datasets

## Drebin_215

[Drebin_215 (ORIGINAL)](https://figshare.com/articles/dataset/Android_malware_dataset_for_machine_learning_2/5854653)

[Drebin_251 (COMPLETO E LIMPO, isto é, pré-processado)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/drebin_215_all.csv)

[Drebin_251_permissions (APENAS PERMISSÕES)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/drebin_215_permissions.csv)

[Drebin_251_api_calls (APENAS CHAMADAS DE API)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/drebin_215_api_calls.csv)

## Androcrawl

[Androcrawl (ORIGINAL)](https://github.com/phretor/ransom.mobi/blob/gh-pages/f/filter.7z)

[Androcrawl (COMPLETO E LIMPO, isto é, pré-processado)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/androcrawl_all.csv)

[Androcrawl_permissions (APENAS PERMISSÕES)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/androcrawl_permissions.csv)

[Androcrawl_api_calls (APENAS CHAMADAS DE API)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/androcrawl_api_calls.csv)

## MD46K

[MD46K (ORIGINAL, COMPLETO)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/md46k_original.csv.zip)

[MD46K (DETALHES DAS AMOSTRAS: SHA256, NOME, PACOTE)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/md46k_SHA256_nome_pacote.csv.zip)

[MD46K (LIMPO, isto é, pré-processado)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/md46k_all.csv.zip)

[MD46K_permissions (APENAS PERMISSÕES)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/md46k_permissions.csv.zip)

[MD46K_api_calls (APENAS CHAMADAS DE API)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/md46k_api_calls.csv.zip)

Passo-a-passo da construção do dataset MD46K:

- *Etapa 1*:  **seleção** dos 46K aplicativos, datados a partir de 2018, do [AndroZoo](https://androzoo.uni.lu/). [Lista dos 46K APKs selecionados (SHA256, NOME, PACOTE)](https://github.com/Malware-Hunter/sbseg22_feature_selection/blob/main/datasets/md46k_SHA256_nome_pacote.csv.zip)

- *Etapa 2*: **download** dos APKs dos 46K aplicativos. O próprio [AndroZoo disponibiliza uma API para o download dos APKs](https://androzoo.uni.lu/api_doc) dos aplicativos.

- *Etapa 3*: **rotulação** das amostras, utilizando a API do serviço online do [VirusTotal](https://www.virustotal.com), que utiliza mais de 60 scanners para analisar e rotular um aplicativo entre benigno ou maligno.

- *Etapa 4*: **extração das características** estáticas dos APKs utilizando a ferramenta [AndroGuard](https://github.com/androguard/androguard).

- *Etapa 5*: **análise das características** utilizando a documentação oficial da Google, outros datasets existentes e, também, outras ferramentas e serviços de extração, como [Malscan](https://github.com/malscan/malscan), [Koodous](https://koodous.com/) e [SandDroid](http://www.sandroid.com/). 

- *Etapa 6*: **construção** do dataset a partir dos dados gerados e validados nas etapas anteriores. O dataset é representado por uma matriz de linhas, que representam as amostras, e colunas, que representam as diversas características das amostras. Cada características (e.g., permissão READ_SMS) está presente (1) ou ausente (0) na amostra. 

- *Etapa 7*: **sanitização** do dataset, isto é, correção de problemas e ruídos nos dados, como registros duplicados, valores incoerentes, formato inadequado dos dados e valores faltantes.
