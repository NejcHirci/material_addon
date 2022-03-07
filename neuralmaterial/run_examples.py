import os
import glob


def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list


epochs = 2000
h_res = 1024
w_res = 1024
seed = 42

root_dir = 'D:/new_materials/NeuralMaterial/'

mat_list = gyListNames(root_dir + '*')
model_path = './trainings/Neuralmaterial'

print(mat_list)
for id, mat in enumerate(mat_list):
    mat_dir = root_dir + mat
    out_dir = mat_dir + '/out/'
    print(id, mat)
    cmd = 'python -u ./scripts/test.py' \
        + ' --model ' + model_path \
        + ' --input_path ' + mat_dir \
        + ' --output_path ' + out_dir \
        + ' --epochs ' + str(epochs) \
        + ' --h ' + str(h_res) \
        + ' --w ' + str(w_res)

    os.system(cmd)
