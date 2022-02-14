import os
import glob

def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list

root_dir = 'materialGAN/data/'
in_dir = 'D:/generated/matgan_generated/'
out_dir = 'D:/generated/matgan_generated/'
cp_dir  = root_dir + 'pretrain/'
vgg_dir = cp_dir + 'vgg_conv.pt'

N = 9
epochs = 2000
epochW = 10
epochN = 10
loss = [1000, 0.001, -1, -1]
lr = 0.02

mat_list = gyListNames(in_dir + 'real_*')

for id, mat in enumerate(mat_list):
    print(id, mat)
    mat_in_dir = os.path.join(in_dir, mat)
    mat_out_dir = os.path.join(out_dir, mat)
    mat_out_dir = os.path.join(mat_out_dir, 'out/')

    cmd = 'python ./materialGAN/src/optim.py' \
        + ' --in_dir ' + mat_in_dir \
        + ' --out_dir ' + mat_out_dir \
        + ' --vgg_weight_dir ' + vgg_dir \
        + ' --num_render_used ' + str(N) \
        + ' --epochs ' + str(epochs) \
        + ' --sub_epochs ' + str(epochW) + ' ' + str(epochN) \
        + ' --loss_weight ' + str(loss[0]) + ' ' + str(loss[1]) + ' ' + str(loss[2]) + ' ' + str(loss[3])\
        + ' --optim_latent' \
        + ' --lr ' + str(lr) \
        + ' --gan_latent_init ' + cp_dir + 'latent_avg_W+_256.pt' \
        + ' --gan_noise_init ' + cp_dir + 'latent_const_N_256.pt' 

    print(cmd)
    os.system(cmd)