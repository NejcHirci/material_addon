import argparse
import sys
from applications import interpolate_single_texture

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Optimization -- GY')
    parser.add_argument('--latent_path', required=True)
    parser.add_argument('--noise_path', required=True)
    parser.add_argument('--save_dir', required=True)

    args = parser.parse_args()

    direct_dir = 'materialGAN/data/directions/'
    mat_list = ['leather_darkbrown', 'metal_rust_blue', 
    'plastic_green', 'plastic_red_carton', 'rocks_brown', 
    'stone_spec_shiny', 'wall_plaster_white', 'wood_beige']

    latent_list = [args.latent_path]
    noises_list = [args.noise_path]

    for mat in mat_list:
        latent_list.append(direct_dir + mat + '/optim_latent.pt')
        noises_list.append(direct_dir + mat + '/optim_noise.pt')

    interpolate_single_texture(latent_list, noises_list, args.save_dir)
