#!/usr/bin/env python3
"""
Train the base model using the learning data

* Prerequisite
    1. Run 06_make_learning_data.py
    2. Run 07_make_model.py
"""
import os
import sys
from settings import PROJECT_DIR


def main():
    # path settings
    data_list_dir = f'{PROJECT_DIR}/data/learning-data-list'
    train_data_list_path = f'{data_list_dir}/train_set_data_paths.txt'
    valid_data_list_path = f'{data_list_dir}/valid_set_data_paths.txt'

    if not os.path.isfile(train_data_list_path) or not os.path.isfile(valid_data_list_path):
        sys.exit(f'[ERROR] The files for lists of learning data does not exist. '
                 f'Check the paths or run 06_make_learning_data.py')

    model_dir = f'{PROJECT_DIR}/model'
    base_model_path = f'{model_dir}/base_model.hdf5'
    train_model_path = f'{model_dir}/train_model.hdf5'

    if not os.path.isfile(base_model_path):
        sys.exit(f'[ERROR] The base model \'{base_model_path}\' does not exist. '
                 f'Check the path or run 07_make_model.py')

    brain_script = f'{PROJECT_DIR}/src/brain.py'  # script for making our model

    # Execution
    cmd = f'{brain_script} train_model {train_model_path} {base_model_path} ' \
          f'{train_data_list_path} {valid_data_list_path};'
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
