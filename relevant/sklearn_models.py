from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import ast
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dtime
from prettytable import PrettyTable

import constants
import ensemble

tqdm.pandas()


def get_classifiers():
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    # classifiers = [
    #     KNeighborsClassifier(3),
    #     SVC(kernel="linear", C=0.025),
    #     SVC(gamma=2, C=1),
    #     GaussianProcessClassifier(1.0 * RBF(1.0)),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     MLPClassifier(alpha=1, max_iter=1000),
    #     AdaBoostClassifier(),
    #     GaussianNB(),
    #     QuadraticDiscriminantAnalysis(),
    # ]

    classifiers = [
        KNeighborsClassifier(1),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    return classifiers, names


def train_models(df_data):
    h = 0.02  # step size in the mesh

    classifiers, names = get_classifiers()

    X_train, X_test, y_train, y_test = split_data(df_data)
    list_records = []

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        y_prob = []
        try:
            y_prob = clf.decision_function(X_train)
        except:
            print(f"{name}: no decision")
        print(f"{name}: \n{y_test}\n{y_pred}\n{y_prob}")
        metrics = get_metrics(y_test, y_pred)

        try:
            rocauc = round(roc_auc_score(y_test, clf.decision_function(X_train)), 2)
            print(rocauc)
        except:
            print(f"rocauc failed")

        # print(f"name: {name}, metrics: {metrics}")
        record = {
            "model": name,
        }
        record.update(metrics)
        list_records.append(record)

    df_eval = pd.DataFrame.from_records(list_records)
    print(df_eval)
    return df_eval


def get_metrics(y_actual, y_pred):
    """
        tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
        (tn, fp, fn, tp)
        (0, 2, 1, 1)
    :param y_actual:
    :param y_pred:
    :return:
    """
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    dict_metrics = {
        'accuracy': round(accuracy_score(y_actual, y_pred), 2),
        'precision': round(precision_score(y_actual, y_pred), 2),
        'recall': round(recall_score(y_actual, y_pred), 2),
        'f1': round(f1_score(y_actual, y_pred), 2),
        'rocauc': round(roc_auc_score(y_actual, y_pred), 2),
        'confusion_tuple': {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }
    return dict_metrics


def split_data(df_data, test_size=0.35, random_state=42):
    columns = df_data.columns
    remove_cols = ["d_id", "label"]
    y = df_data["label"].tolist()
    columns = [item for item in columns if item not in remove_cols]
    x = df_data[columns]
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def fix_row(text):
    return ast.literal_eval(text)


def fix_np(text):
    return np.fromstring(text[1:-1], dtype=np.float, sep=' ')


def fix_list(df_data, cols):
    for col in cols:
        print(col)
        df_data[col] = df_data.progress_apply(lambda row: fix_np(row[col]), axis=1)
    return df_data


def alt_run():
    out_path = os.path.join(constants.DIR_DATA, f"df_ensemble.csv")
    print(f"Reading ensemble dataset from: {out_path}")

    df_ensemble = pd.read_csv(out_path)
    cols = ["T2w_fc1", "T2w_dense_2",
            "T1wCE_dense_3", "T1wCE_dense_2",
            "FLAIR_fc1", "FLAIR_dense_2"]
    df_ensemble = fix_list(df_ensemble, cols)
    df_ensemble = ensemble.flatten_cols(df_ensemble)

    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_ensemble_flat.csv")
    df_ensemble.to_csv(out_path, index=False)

    df_eval = train_models(df_ensemble)

    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_eval_1.csv")
    print(f"Writing ensemble evaluation to: {out_path}")
    df_eval.to_csv(out_path, index=False)


def main():
    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_ensemble_std.csv")
    print(f"Reading ensemble dataset from: {out_path}")

    ensemble.create_ensembled_features()

    df_data = pd.read_csv(out_path)
    df_eval = train_models(df_data)

    str_ts = dtime.datetime.now().strftime(constants.ts_fmt)
    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_eval_{str_ts}.csv")
    print(f"Writing ensemble evaluation to: {out_path}")
    df_eval.to_csv(out_path, index=False)

    # alt_run()


if __name__ == '__main__':
    main()
