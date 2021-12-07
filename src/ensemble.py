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
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    features = extractor(data)
    return features


def get_intermediate_output_2(model, data, layer_name='my_layer'):
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(data)
    return intermediate_output


def get_varied_features(list_models, list_model_paths, data, list_data_paths):
    list_all_features = []
    layer_name = {"3dcnn": "dense_2", "resnet": "dense_2"}
    layer_name = {"3dcnn": "dense_3", "resnet": "fc1"}

    for d_index, dict_data in enumerate(list_data_paths):
        dict_tmp = {
            "data_id": list_data_paths[d_index]["data_id"],
            "label": list_data_paths[d_index]["label"]
        }
        for index, model in enumerate(list_models):
            dense_3 = get_intermediate_output_1(model, data, layer_name="dense_3")
            dense_2 = get_intermediate_output_1(model, data, layer_name="dense_2")
            dict_tmp["dense_3"] = dense_3
            dict_tmp["dense_2"] = dense_2
            dict_tmp["mri_type"] = list_model_paths[index]["mri_type"]

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
    model = keras.models.load_model(model_path)
    print(model.summary())
    return model


def load_models(list_paths):
    list_models = []
    for index, dict_model_path in enumerate(list_paths):
        model = _load_model(dict_model_path["path"])
        list_models.append(model)

    return list_models


def get_data(list_data):
    patient_scans_test = np.array([utils.process_scan(path) for path in list_data])
    return patient_scans_test


def get_list_model_paths():
    list_model_paths = [
        {
            "path": "models/3d_image_classification_simple_T1w_1.h5",
            "model_type": "3dcnn",
            "mri_type": "T1w"
        },
        {
            "path": "models/3d_image_classification_simple_T2w_1.h5",
            "model_type": "3dcnn",
            "mri_type": "T2w"
        }
    ]
    return list_model_paths


def get_list_data_paths():
    list_data_paths = [
        {
            "data_id": "00001",
            "path": "models/3d_image_classification_simple_T1w_1.h5",
            "label": 0
        },
        {
            "data_id": "00001",
            "path": "models/3d_image_classification_simple_T1w_1.h5",
            "label": 0
        },
        {
            "data_id": "00001",
            "path": "models/3d_image_classification_simple_T2w_1.h5",
            "label": 1
        }
    ]
    return list_data_paths


def create_ensembled_features(args):
    list_model_paths = get_list_model_paths()
    list_data_paths = get_list_data_paths()

    list_models = load_models(list_model_paths)
    data = get_data(list_data_paths)
    list_records = get_varied_features(list_models, list_model_paths, data, list_data_paths)

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
