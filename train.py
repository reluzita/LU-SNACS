from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
import pandas as pd
import argparse

RF_CLF = RandomForestClassifier(n_estimators=10)
BRF_CLF = BaggingClassifier(base_estimator=RF_CLF, n_estimators=10)

def get_us(us_strategy, ratio):
    if us_strategy == 'random':
        return RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    elif us_strategy == 'nearmiss1':
        return NearMiss(sampling_strategy=ratio, version=1)
    elif us_strategy == 'nearmiss2':
        return NearMiss(sampling_strategy=ratio, version=2)
    elif us_strategy == 'nearmiss3':
        return NearMiss(sampling_strategy=ratio, version=3)

def train_us_ratios(X_train, y_train, X_test, y_test, ratios, us_strategy, results: dict):
    for ratio in ratios:
        print(f"----RATIO:{ratio}----")
        undersample = get_us(us_strategy, ratio)

        # transform the dataset
        X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
        clf = BRF_CLF.fit(X_train_us, y_train_us)
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
        results["weighted avg precision"].append(report['weighted avg']['precision'])
        results["weighted avg recall"].append(report['weighted avg']['recall'])
        results["weighted avg f1"].append(report['weighted avg']['f1-score'])

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_roc = auc(fpr, tpr)

        results["auc"].append(auc_roc)
        results["fpr"].append(list(fpr))
        results["tpr"].append(list(tpr))
    
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'train.py',
        description = 'Construct dataset for supervised link prediction from a network dataset')

    parser.add_argument('datafile', type=str, help='Dataset file name')
    parser.add_argument('us_strategy', type=str, help='Undersampling technique to apply', choices=['random', 'nearmiss1', 'nearmiss2', 'nearmiss3', 'tomek'])
    parser.add_argument('-b', '--baseline', action='store_true')

    args = parser.parse_args()

    print("Reading file...")
    data_file = args.datafile
    features = pd.read_csv('data/clean_datasets/' + data_file).set_index('Unnamed: 0')

    feature_names = list(features.columns)
    feature_names.remove('label')
    X_train, X_test, y_train, y_test = train_test_split(features[feature_names].values, features['label'], test_size=0.3, random_state=0)
    
    results = {
        'majority precision': [],
        'majority recall': [],
        'majority f1': [],
        'minority precision': [],
        'minority recall': [],
        'minority f1': [],
        'accuracy': [],
        'macro avg precision': [],
        'macro avg recall': [],
        'macro avg f1': [],
        'weighted avg precision': [],
        'weighted avg recall': [],
        'weighted avg f1': [],
        "auc": [],
        "fpr": [],
        "tpr": [],
    }

    if args.baseline:
        print("Establishing baseline predictions...")
        clf = BRF_CLF.fit(X_train, y_train)
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
            'macro avg f1': [report['macro avg']['f1-score']],
            'weighted avg precision': [report['weighted avg']['precision']],
            'weighted avg recall': [report['weighted avg']['recall']],
            'weighted avg f1': [report['weighted avg']['f1-score']]
        }

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_roc = auc(fpr, tpr)

        results["auc"] = [auc_roc]
        results["fpr"] = [list(fpr)]
        results["tpr"] = [list(tpr)]

    original_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]

    print("Training with varying ratios...")
    
    ratios = [r for r in [0.2, 0.4, 0.6, 0.8, 1.0] if r > original_ratio]
    results = train_us_ratios(X_train, y_train, X_test, y_test, ratios, args.us_strategy, results)

    ratios = [original_ratio] + ratios
    results['ratio'] = ratios

    results = pd.DataFrame(results)

    filename = data_file.split('.')[0]
    results.set_index('ratio').to_csv(f'results/{filename}_{args.us_strategy}_results.csv')