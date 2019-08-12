#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Train the base model using the learning data

* Prerequisite
    1. Run 08_make_learning_data.py
    2. Run 09_make_model.py
"""
import os
import sys


def main():
    # path settings
    project_dir = '/extdata4/baeklab/minwoo/projects/deep-purity'
    data_list_dir = f'{project_dir}/data/learning-data-list'
    train_data_list_path = f'{data_list_dir}/train_set_data_paths.txt'
    valid_data_list_path = f'{data_list_dir}/valid_set_data_paths.txt'

    if not os.path.isfile(train_data_list_path) or not os.path.isfile(valid_data_list_path):
        sys.exit(f'[ERROR] The files for lists of learning data does not exist. '
                 f'Check the paths or run 08_make_learning_data.py')

    model_dir = f'{project_dir}/model'
    base_model_path = f'{model_dir}/base_model_190809.hdf5'
    train_model_path = f'{model_dir}/train_model_190809.hdf5'

    if not os.path.isfile(base_model_path):
        sys.exit(f'[ERROR] The base model \'{base_model_path}\' does not exist. '
                 f'Check the path or run 09_make_model.py')

    brain_script = f'{project_dir}/src/brain.py'  # script for making our model

    # Execution
    cmd = f'{brain_script} train_model {train_model_path} {base_model_path} ' \
          f'{train_data_list_path} {valid_data_list_path};'
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
