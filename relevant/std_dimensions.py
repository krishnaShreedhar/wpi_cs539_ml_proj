import pandas as pd
from collections import Counter
import os


def load_all():
    list_paths = ["../data/dimensions/FLAIR.csv", "../data/dimensions/T1w.csv",
                  "../data/dimensions/T1wCE.csv", "../data/dimensions/T2w.csv"]

    cols = ["Filepath", "Shape", "dim1", "dim2", "dim3"]
    list_df = [{"mri_type": path.split('/')[-1], "df": pd.read_csv(path)[cols]} for path in list_paths]

    return list_df


def get_counts(dict_df):
    shape_counts = dict_df["df"]["Shape"].value_counts()
    print(shape_counts)


def filter(list_dict_dfs):
    print(f"-------------------------")
    for dict_df in list_dict_dfs:
        df = dict_df["df"]
        cond_dim1 = df["dim1"] >= 192
        cond_dim2 = df["dim2"] >= 256
        cond_dim3 = df["dim3"] >= 60

        cond12 = cond_dim1 & cond_dim2
        cond123 = cond12 & cond_dim3

        df_filtered = df[cond123]
        print(df_filtered.describe())
        print(f"{len(df_filtered.index)}\n{df_filtered.head}")
        base = "../outputs/dimensions/"
        if not os.path.exists(base):
            os.makedirs(base)
        path = os.path.join(base, dict_df["mri_type"])
        df_filtered.to_csv(path, index=False)


def main():
    list_df = load_all()
    for dict_df in list_df:
        print(dict_df["mri_type"])
        print(dict_df["df"].head())
        print(dict_df["df"].describe())
        get_counts(dict_df)

    print(f"-------------------------")

    filter(list_df)


if __name__ == '__main__':
    main()
