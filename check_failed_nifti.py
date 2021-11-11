import os

nifti_dir = "Nifti/"
patient_list = os.listdir(nifti_dir)
mri_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']

for patient in patient_list:
    for mri_type in mri_types:
        files = os.listdir(nifti_dir + patient + "/" + mri_type + "/")
        if len(files) == 0:
            print("Failed patient: ", patient)
            print("Failed scan type: ", mri_type)
            print()
            print()


# Failed patients output from Turing:

# Failed patient:  00148
# Failed scan type:  T1wCE
#
# Failed patient:  00108
# Failed scan type:  T2w
#
# Failed patient:  00524
# Failed scan type:  T2w
#
# Failed patient:  00147
# Failed scan type:  T2w
#
# Failed patient:  00120
# Failed scan type:  T2w
#
# Failed patient:  00132
# Failed scan type:  T2w
#
# Failed patient:  01010
# Failed scan type:  T1wCE
#
# Failed patient:  00137
# Failed scan type:  T2w
#
# Failed patient:  00143
# Failed scan type:  T2w
#
# Failed patient:  00107
# Failed scan type:  T2w
#
# Failed patient:  00834
# Failed scan type:  T1wCE

