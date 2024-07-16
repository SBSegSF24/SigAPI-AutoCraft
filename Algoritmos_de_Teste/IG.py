import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import argparse
import sys
from pathlib import Path

# Define the path to the directory
directory_path = Path('../tcc_laura_2022/PCA vs InfoGain')

def get_X_y(args, dataset):
    if(args.class_column not in dataset.columns):
        print(f'Class Column {args.class_column} Not Found in Dataset {args.dataset}')
        return None, None
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def calculateMutualInformationGain(features, target):
    features_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {'features': features_names, 'score': mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by=['score'], ascending=False)
    return df

def mi_features(dataset, args):
    X_dataset, y_dataset = get_X_y(args, dataset)
    mig = calculateMutualInformationGain(X_dataset, y_dataset)
    mig.index = range(len(mig.index))
    #Mostra todos os dados do dataframe
    #for linha in range(len(mig.index)):
    #    print(mig['features'][linha], mig['score'][linha])
    #print(args.dataset)

    # Ensure the directory exists
    directory_path.mkdir(parents=True, exist_ok=True)

    # Salvar os scores ordenados em um arquivo CSV
    output_file_path = directory_path / f'IG_{Path(args.dataset).stem}.csv'
    mig.to_csv(output_file_path, index=False)

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar='str',
        help='Dataset (csv file).', type=str, required=True)
    parser.add_argument(
        '--class-column', type = str,
        default = 'class', metavar = 'CLASS_COLUMN',
        help = 'Class Column ID. Default: class')
    args = parser.parse_args(argv)
    return args

if __name__=="__main__":
    args = parse_args(sys.argv[1:])
    try:
        dataset = pd.read_csv(args.dataset)
    except BaseException as e:
        print(e)
        exit(1)

    mi_features(dataset, args)
