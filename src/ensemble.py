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


def get_varied_features(list_models, data):
    features = []
    for model in list_models:
        temp_features = model.predict(data)
        features.append(temp_features)
    return features


# def get_features():
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
    # TODO: Add load model code
    model = keras.models.load_model("my_h5_model.h5")
    print(model.summary())
    return model


def load_models(list_paths):
    list_models = []
    for index, model_path in enumerate(list_paths):
        model = _load_model(model_path)
        list_models.append(model)

    return list_models


def create_ensembled_features(args):
    list_paths = ["../models/3d_image_classification_lr1e1_softmax_batch1_rotate.h5"]
    list_models = load_models(list_paths)

    list_data_paths = ["../outputs/Nifti/T1w/.nii"]
    data = np.array([utils.process_scan(path) for path in list_data_paths])
    list_records = get_varied_features(list_models, data)

    df_ensemble = pd.DataFrame.from_records(list_records)

    out_path = os.path.join(constants.DIR_OUTPUTS, f"df_ensemble.csv")
    df_ensemble.to_csv(out_path, index=False)


def train_ensembled_models(args):
    list_models = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("r")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    create_ensembled_features(args)


if __name__ == '__main__':
    main()
