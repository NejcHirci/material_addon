import os
import glob

def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list


in_dir = 'D:/new_materials/MaterialGAN/'

mat_list = gyListNames(in_dir + '*')

for id, mat in enumerate(mat_list):
    print(id, mat)
    mat_dir = os.path.join(in_dir, mat)

    mat_in_dir = mat_dir
    mat_out_dir = os.path.join(mat_dir, 'input/')

    if not os.path.exists(mat_out_dir):
        cmd = 'python ./materialGAN/tools/generate_inputs.py' \
            + ' --in_dir ' + mat_in_dir \
            + ' --out_dir ' + mat_out_dir \

        print(cmd)
        os.system(cmd)
    else:
        print(mat +" already generated")