#!/home/sonic/baeklab/Hyeonseong/anaconda3/envs/minwoo/bin/python
"""
Test the trained model by predicting purities of the test data

* Prerequisite
    1. Run 03_make_test_data.py
"""
import os
import sys
from settings import PROJECT_DIR


def main():
    print(f'[LOG] DeepPurity model testing', file=sys.stderr)

    # param settings
    cell_lines = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depth = '30x'
    model_ver = '190911-cnn1'

    # path settings
    test_data_list_dir = f'{PROJECT_DIR}/data/test-data-2-list'
    result_dir = f'{PROJECT_DIR}/results/prediction/{model_ver}'
    os.makedirs(result_dir, exist_ok=True)

    model_dir = f'{PROJECT_DIR}/model'
    train_model_path = f'{model_dir}/train_model_{model_ver}.hdf5'

    if not os.path.isfile(train_model_path):
        sys.exit(f'[ERROR] The base model \'{train_model_path}\' does not exist. '
                 f'Check the path or run training/10_train_model.py')

    brain_script = f'{PROJECT_DIR}/src/brain.py'  # script for making our model

    for cell_line in cell_lines:
        print(f'[LOG] --- Cell line: {cell_line}', file=sys.stderr)
        # in-loop path settings
        test_data_list_path = f'{test_data_list_dir}/{cell_line}_{depth}_data_paths.txt'
        result_path = f'{result_dir}/result_{cell_line}_{depth}.txt'

        if not os.path.isfile(test_data_list_path):
            sys.exit(f'[ERROR] The files for lists of learning data does not exist. '
                     f'Check the paths or run 03_make_test_data.py')

        # Execution
        cmd = f'{brain_script} test_model {result_path} {train_model_path} {test_data_list_path};'
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()
