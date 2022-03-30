"""SeFA. Modified by: @NejcHirci"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import sys
import copy
sys.path.insert(1, 'materialGAN/src/')
import global_var
from applications import save_render_and_map, lerp_noises
sys.path.insert(1, 'materialGAN/higan/models/')
from stylegan2_generator import StyleGAN2Generator

if __name__ == '__main__':
    """ Parse arguments. """
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--latent_path', type=str, required=True)
    parser.add_argument('--noise_path', type=str, required=True)
    parser.add_argument('--start_distance', type=float, default=-10.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=10.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=2,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')

    args = parser.parse_args()

    # Seed same
    np.random.seed(42)
    torch.manual_seed(42)

    # Get Factorized weights
    global_var.init_global_noise(256, 'materialGAN/data/pretrain/latent_const_N_256.pt')
    generator = StyleGAN2Generator('svbrdf')
    boundaries = np.load('sefa/directions/boundaries_0-13.npy')
    values = np.load('sefa/directions/values_0-13.npy')

    device=torch.device('cpu')
    if torch.cuda.is_available:
        device=torch.device('cuda')

    # Already in wp+ space
    code = torch.load(args.latent_path).detach().cpu().numpy()

    noises_list = torch.load(args.noise_path, map_location=device)
    noises = []
    for noise in noises_list:
        if torch.cuda.is_available():
            noise = noise.cuda()
        noises.append(noise)

    input_path = os.path.abspath(os.path.join(args.latent_path, os.pardir, os.pardir))
    input_path = os.path.join(input_path, 'input/')


    def rand_noise():
        global_var.init_global_noise(256, "random")
        return copy.deepcopy(global_var.noises)
    rand_noises = [rand_noise() for _ in range(8)]
    
    for sem_id in range(8):
        boundary = boundaries[sem_id:sem_id + 1]
        distances = [-3.0, 3.0]
        for col_id, d in enumerate(distances, start=1):
            temp_code = code.copy()
            temp_code[:, 0:13, :] += boundary * d
            global_var.noises = lerp_noises(noises, rand_noises[sem_id], 0.3 * d/abs(d))
            torch.save(torch.from_numpy(temp_code).type(torch.FloatTensor), os.path.join(args.save_dir, f'{sem_id}_{col_id}_optim_latent.pt'))
            torch.save(global_var.noises, os.path.join(args.save_dir, f'{sem_id}_{col_id}_optim_noise.pt'))
            image = generator.net.synthesis(torch.from_numpy(temp_code).type(torch.FloatTensor).cuda())
            image = generator.get_value(image)[0]
            save_render_and_map(f"{sem_id}_{col_id}", args.save_dir, image, input_path)
        print("Semantics generated {}/8".format(sem_id+1))
