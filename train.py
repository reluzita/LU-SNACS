from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
import sys
import pandas as pd
import argparse

def get_us(us_strategy, ratio):
    if us_strategy == 'random':
        return RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    elif us_strategy == 'nearmiss1':
        return NearMiss(sampling_strategy=ratio, version=1)
    elif us_strategy == 'nearmiss2':
        return NearMiss(sampling_strategy=ratio, version=2)
    elif us_strategy == 'nearmiss3':
        return NearMiss(sampling_strategy=ratio, version=3)

def train_us_ratios(X_train, y_train, X_test, y_test, ratios, us_strategy, results):
    for ratio in ratios:
        # print(f"----RATIO:{ratio}----")
        undersample = get_us(us_strategy, ratio)

        # transform the dataset
        X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
        clf = LogisticRegression().fit(X_train_us, y_train_us)
        y_pred = clf.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results["majority precision"].append(report['0']['precision'])
        results["majority recall"].append(report['0']['recall'])
        results["majority f1"].append(report['0']['f1-score'])
        results["minority precision"].append(report['1']['precision'])
        results["minority recall"].append(report['1']['recall'])
        results["minority f1"].append(report['1']['f1-score'])
        results["accuracy"].append(report['accuracy'])
        results["macro avg precision"].append(report['macro avg']['precision'])
        results["macro avg recall"].append(report['macro avg']['recall'])
        results["macro avg f1"].append(report['macro avg']['f1-score'])
    
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Extract data',
        description = 'Construct dataset for supervised link prediction from a network dataset')

    parser.add_argument('datafile', type=str, help='Dataset file name')
    parser.add_argument('us_strategy', type=str, help='Undersampling technique to apply', choices=['random', 'nearmiss1', 'nearmiss2', 'nearmiss3', 'tomek'])
    
    args = parser.parse_args()

    data_file = args.datafile
    features = pd.read_csv('data/clean_datasets/' + data_file).set_index('Unnamed: 0')

    feature_names = list(features.columns)
    feature_names.remove('label')
    X_train, X_test, y_train, y_test = train_test_split(features[feature_names].values, features['label'], test_size=0.3, random_state=0)
   
    clf = LogisticRegression()
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    results = {
        'majority precision': [report['0']['precision']],
        'majority recall': [report['0']['recall']],
        'majority f1': [report['0']['f1-score']],
        'minority precision': [report['1']['precision']],
        'minority recall': [report['1']['recall']],
        'minority f1': [report['1']['f1-score']],
        'accuracy': [report['accuracy']],
        'macro avg precision': [report['macro avg']['precision']],
        'macro avg recall': [report['macro avg']['recall']],
        'macro avg f1': [report['macro avg']['f1-score']]
    }

    original_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]

    if args.us_strategy == 'tomek':
        undersample = TomekLinks(sampling_strategy='majority')
        X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
        clf = LogisticRegression().fit(X_train_us, y_train_us)
        y_pred = clf.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results["majority precision"].append(report['0']['precision'])
        results["majority recall"].append(report['0']['recall'])
        results["majority f1"].append(report['0']['f1-score'])
        results["minority precision"].append(report['1']['precision'])
        results["minority recall"].append(report['1']['recall'])
        results["minority f1"].append(report['1']['f1-score'])
        results["accuracy"].append(report['accuracy'])
        results["macro avg precision"].append(report['macro avg']['precision'])
        results["macro avg recall"].append(report['macro avg']['recall'])
        results["macro avg f1"].append(report['macro avg']['f1-score'])

        ratios = [y_train_us.value_counts()[1] / y_train_us.value_counts()[0]]

    else:
        ratios = [r for r in [0.2, 0.4, 0.6, 0.8, 1.0] if r > original_ratio]
        results = train_us_ratios(X_train, y_train, X_test, y_test, ratios, args.us_strategy, results)

    ratios = [y_train.value_counts()[1] / y_train.value_counts()[0]] + ratios
    results['ratio'] = ratios

    results = pd.DataFrame(results)
    for col in results.columns:
        results[col] = results[col].map('{:,.3f}'.format)

    filename = data_file.split('.')[0]
    results.set_index('ratio').to_csv(f'results/{filename}_{args.us_strategy}_results.csv')