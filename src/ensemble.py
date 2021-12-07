import argparse
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np

import utils
import constants


def get_intermediate_output_1(model, data, layer_name='my_layer'):
    input_data = np.expand_dims(data, axis=0)
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    features = extractor(input_data)
    return features


def get_intermediate_output_2(model, data, layer_name='my_layer'):
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(data)
    return intermediate_output


def get_varied_features(list_models, list_model_paths, list_scans):
    list_all_features = []
    layer_name = {"3dcnn": "dense_2", "resnet": "dense_2"}
    layer_name = {"3dcnn": "dense_3", "resnet": "fc1"}

    for d_index, dict_data in enumerate(list_scans):
        dict_tmp = {
            "d_id": dict_data["data_id"],
            "d_path": dict_data["path"],
            "label": dict_data["label"]
        }
        for m_index, model in enumerate(list_models):
            mri_type = list_model_paths[m_index]["mri_type"]
            npa_data = dict_data["scans"][mri_type]

            if npa_data is not None:
                layer_name = "dense_3"
                layer_output = get_intermediate_output_1(model, npa_data, layer_name=layer_name)
                dict_tmp[f"{mri_type}_{layer_name}"] = layer_output

                layer_name = "dense_2"
                layer_output = get_intermediate_output_1(model, npa_data, layer_name=layer_name)
                dict_tmp[f"{mri_type}_{layer_name}"] = layer_output

        list_all_features.append(dict_tmp)
    return list_all_features


# def get_features(model, data):
#     model = Sequential([
#         layers.Conv2D(32, 3, activation='relu'),
#         layers.Conv2D(32, 3, activation='relu'),
#         layers.MaxPooling2D(2),
#         layers.Conv2D(32, 3, activation='relu'),
#         layers.Conv2D(32, 3, activation='relu'),
#         layers.GlobalMaxPooling2D(),
#         layers.Dense(10),
#     ])
#     list_models = [model, model]
#     data = []
#     get_varied_features(list_models, data)


def _load_model(model_path, **kwargs):
    list_str_summary = []
    model = keras.models.load_model(model_path)
    model.summary(print_fn=lambda x: list_str_summary.append(f"{x}"))
    print('\n'.join(list_str_summary))
    return model


def load_models(list_paths):
    list_models = []
    for index, dict_model_path in enumerate(list_paths):
        model = _load_model(dict_model_path["path"])
        list_models.append(model)

    return list_models


def _get_scans(patient_id, label):
    mri_types = constants.mri_types
    dict_scans = dict()
    status = True
    for mri_type in mri_types:
        path = f"../data/project_folder_correct_{mri_type}/" \
               f"cross_val_folds/fold_3/train/{label}/{str(patient_id).zfill(5)}.nii"
        try:
            dict_scans[mri_type] = utils.process_scan(path)
        except:
            status = False
            print(f"Failed: Patient: {patient_id}, Scan: {mri_type}, Label: {label}")
            dict_scans[mri_type] = None

    dict_scans["status"] = status

    return dict_scans


def get_all_data(list_data_paths):
    list_scans = []
    for dict_data in list_data_paths:
        dict_scans = {
            "d_id": dict_data["d_id"],
            "label": dict_data["label"],
            "scans": _get_scans(dict_data["d_id"], dict_data["label"])
        }

        list_scans.append(dict_scans)
    return list_scans


def get_list_model_paths():
    list_model_paths = [
        {
            "m_id": "3dcnn_T1w_1",
            "path": "../models/3d_image_classification_simple_T1w_1.h5",
            "model_type": "3dcnn",
            "mri_type": "T1w"
        },
        {
            "m_id": "3dcnn_T2w_1",
            "path": "../models/3d_image_classification_simple_T2w_1.h5",
            "model_type": "3dcnn",
            "mri_type": "T2w"
        }
    ]
    return list_model_paths


def get_list_data_paths():
    list_data_paths = [
        {
            "d_id": 9,
            "label": 0
        },
        {
            "d_id": 444,
            "label": 0
        },
        {
            "d_id": 2,
            "label": 1
        }
    ]
    return list_data_paths


def create_ensembled_features(args):
    list_model_paths = get_list_model_paths()
    list_data_paths = get_list_data_paths()

    list_models = load_models(list_model_paths)
    list_scans = get_all_data(list_data_paths)
    list_records = get_varied_features(list_models, list_model_paths, list_scans)

    df_ensemble = pd.DataFrame.from_records(list_records)

    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_ensemble.csv")
    df_ensemble.to_csv(out_path, index=False)


def train_ensembled_models(args):
    list_models = []


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("r")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    create_ensembled_features(args)


if __name__ == '__main__':
    main()
