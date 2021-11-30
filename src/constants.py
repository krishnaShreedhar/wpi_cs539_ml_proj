import os

DIR_DATA = "../data/"
DIR_OUTPUTS = "../outputs/"
DIR_MODELS = "../models/"

if not os.path.exists(DIR_OUTPUTS):
    os.makedirs(DIR_OUTPUTS)

ts_fmt = "%Y%m%d_%H%M%S"
