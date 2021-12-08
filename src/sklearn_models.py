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

import os
import pandas as pd
from prettytable import PrettyTable

import constants
import ensemble


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
        metrics = get_metrics(y_test, y_pred)

        try:
            rocauc = round(roc_auc_score(y_test, y_pred), 2)
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


def main():
    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_ensemble.csv")
    print(f"Reading ensemble dataset from: {out_path}")

    # ensemble.create_ensembled_features()

    df_data = pd.read_csv(out_path)
    df_eval = train_models(df_data)

    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_eval.csv")
    print(f"Writing ensemble evaluation to: {out_path}")
    df_eval.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
