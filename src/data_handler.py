import pandas as pd
import os
from tqdm import tqdm

import constants

tqdm.pandas()


class MRIDataHandler:

    def __init__(self, all_data_path, base_path="../../../new_data/"):
        self.csv_path = all_data_path
        self.df_data = pd.read_csv(all_data_path)
        self.extract_id_from_path()
        self.base_path = base_path
        self.mri_types = constants.mri_types

    @staticmethod
    def _extract_id_from_path(path):
        basename = os.path.basename(path)
        id = int(basename.split('.')[0])
        return id

    def extract_id_from_path(self):
        print(f"Extracting IDs:")
        self.df_data['d_id'] = self.df_data.progress_apply(lambda row: self._extract_id_from_path(row['filepath']),
                                                           axis=1)

    def _get_row_from_id(self, patient_id, return_dict=True):
        cond_id = self.df_data["d_id"] == patient_id
        row = self.df_data[cond_id]
        if return_dict:
            row = row.to_dict()
        return row

    def get_label(self, patient_id):
        gt_label = self._get_row_from_id(patient_id)['gt_label']
        return gt_label

    @staticmethod
    def get_str_patient_id(patient_id):
        str_id = str(patient_id).zfill(5)
        return str_id

    def get_mri_path(self, patient_id, mri_type):
        if mri_type not in self.mri_types:
            raise ValueError("Wrong mri_type")
        str_id = self.get_str_patient_id(patient_id)
        mri_path = os.path.join(self.base_path, mri_type, f"{str_id}.nii")
        return mri_path

    def
