#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Test the trained model by predicting purities of the test data

* Prerequisite
    1. Run 01_make_test_data.py
"""
import os
import sys


def main():
    print(f'[LOG] DeepPurity model testing', file=sys.stderr)

    # param settings
    cell_lines = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depth = '30x'

    # path settings
    project_dir = '/extdata4/baeklab/minwoo/projects/deep-purity'
    test_data_list_dir = f'{project_dir}/data/test-data-list'
    result_dir = f'{project_dir}/results/prediction'
    os.makedirs(result_dir, exist_ok=True)

    model_dir = f'{project_dir}/model'
    train_model_path = f'{model_dir}/train_model_prev.hdf5'

    if not os.path.isfile(train_model_path):
        sys.exit(f'[ERROR] The base model \'{train_model_path}\' does not exist. '
                 f'Check the path or run training/08_train_model.py')

    brain_script = f'{project_dir}/src/brain.py'  # script for making our model

    for cell_line in cell_lines:
        print(f'[LOG] --- Cell line: {cell_line}', file=sys.stderr)
        # in-loop path settings
        test_data_list_path = f'{test_data_list_dir}/{cell_line}_{depth}_data_paths.txt'
        result_path = f'{result_dir}/result_{cell_line}_{depth}.txt'

        if not os.path.isfile(test_data_list_path):
            sys.exit(f'[ERROR] The files for lists of learning data does not exist. '
                     f'Check the paths or run 01_make_test_data.py')

        # Execution
        cmd = f'{brain_script} test_model {result_path} {train_model_path} {test_data_list_path};'
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()
