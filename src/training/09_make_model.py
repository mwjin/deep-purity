#!/home/sonic/baeklab/Hyeonseong/anaconda3/envs/minwoo/bin/python
"""
Make a command to make a base model for training and execute it
"""
import os


def main():
    # path settings
    project_dir = '/extdata1/baeklab/minwoo/projects/deep-purity'
    model_dir = f'{project_dir}/model'
    base_model_path = f'{model_dir}/base_model_190911-cnn1.hdf5'
    brain_script = f'{project_dir}/src/brain.py'  # script for making our model
    os.makedirs(model_dir, exist_ok=True)

    # Execution
    cmd = f'{brain_script} make_base_model {base_model_path};'
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    main()
