from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import sys
import pandas as pd

FEATURES = ['indegree_i', 'outdegree_i', 'indegree_j', 'outdegree_j', 'common_neighbors',
            'adamic_adar', 'pref_attach', 'jaccard', 'katz_i', 'katz_j']

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python train.py <data_file> <directed|undirected> <undersampling_strategy>")
        exit(1)

    data_file = sys.argv[1]
    features = pd.read_csv('data/clean_datasets/' + data_file)

    if sys.argv[2] == 'directed':
        X_train, X_test, y_train, y_test = train_test_split(features[FEATURES].values, features['label'], test_size=0.3, random_state=0)
    elif sys.argv[2] == 'undirected':
        X_train, X_test, y_train, y_test = train_test_split(features[FEATURES].values, features['label'], test_size=0.3, random_state=0)
    else:
        print("Usage: python train.py <data_file> <directed|undirected> <undersampling_strategy>")
        exit(1)

    clf = LogisticRegression()
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    maj_precision = [report['0']['precision']]
    maj_recall = [report['0']['recall']]
    maj_f1 = [report['0']['f1-score']]
    min_precision = [report['1']['precision']]
    min_recall = [report['1']['recall']]
    min_f1 = [report['1']['f1-score']]
    accuracy = [report['accuracy']]
    precision = [report['weighted avg']['precision']]
    recall = [report['weighted avg']['recall']]
    f1 = [report['weighted avg']['f1-score']]

    original_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]
    ratios = [r for r in [0.2, 0.4, 0.6, 0.8, 1.0] if r > original_ratio]

    us_strategy = sys.argv[3]

    for ratio in ratios:
        # print(f"----RATIO:{ratio}----")
        if us_strategy == 'random':
            undersample = RandomUnderSampler(sampling_strategy=ratio)
        elif us_strategy == 'nearmiss1':
            undersample = NearMiss(sampling_strategy=ratio, version=1)
        elif us_strategy == 'nearmiss2':
            undersample = NearMiss(sampling_strategy=ratio, version=2)
        elif us_strategy == 'nearmiss3':
            undersample = NearMiss(sampling_strategy=ratio, version=3)
        else:
            print("<undersampling_strategy> must be one of: random, nearmiss1, nearmiss2, nearmiss3")
            exit(1)
        # transform the dataset
        X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
        clf = LogisticRegression().fit(X_train_us, y_train_us)
        y_pred = clf.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)

        maj_precision.append(report['0']['precision'])
        maj_recall.append(report['0']['recall'])
        maj_f1.append(report['0']['f1-score'])
        min_precision.append(report['1']['precision'])
        min_recall.append(report['1']['recall'])
        min_f1.append(report['1']['f1-score'])
        accuracy.append(report['accuracy'])
        precision.append(report['weighted avg']['precision'])
        recall.append(report['weighted avg']['recall'])
        f1.append(report['weighted avg']['f1-score'])
    
    ratios = [y_train.value_counts()[1] / y_train.value_counts()[0]] + ratios
    results = pd.DataFrame({
        'ratio': ratios,
        'majority precision': maj_precision,
        'majority recall': maj_recall,
        'majority f1': maj_f1,
        'minority precision': min_precision,
        'minority recall': min_recall,
        'minority f1': min_f1,
        'accuracy': accuracy,
        'weighted avg precision': precision,
        'weighted avg recall': recall,
        'weighted avg f1': f1
    }).set_index('ratio')

    results.index = results.index.map('{:,.3f}'.format)
    for col in results.columns:
        results[col] = results[col].map('{:,.3f}'.format)

    filename = data_file.split('.')[0]
    results.to_csv(f'results/{filename}_{us_strategy}_results.csv')