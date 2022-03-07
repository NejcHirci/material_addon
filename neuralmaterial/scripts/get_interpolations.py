import argparse
from omegaconf import OmegaConf
import torch
from torch.serialization import load
import torchvision.io as io
import torchvision.transforms as transforms
from pathlib import Path
from glob import glob
import sys
import time
sys.path.insert(0, './')

from scripts.test import save_png
from lib.core.utils import seed_everything
from lib.core.trainer import Trainer
from lib.main import NeuralMaterial

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve BRDF decomposition from stationary flash image.')
    parser.add_argument('--model', type=str, required=True,
                        help='yaml config file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='yaml config file and flash images')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='path to weight file')
    parser.add_argument('--gpu', type=bool, required=False, default=True,
                        help='use GPU / CPU')
    parser.add_argument('--h', type=int, required=False, default=384,
                        help='output height')
    parser.add_argument('--w', type=int, required=False, default=512,
                        help='output width')

    args = parser.parse_args()

    device = 'cpu'
    if torch.cuda.is_available() and args.gpu:
        device = 'cuda'

    # load config
    cfg = OmegaConf.load(str(Path(args.model, '.hydra', 'config.yaml')))
    seed_everything(cfg.seed)

    def load_image(in_path):

        tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(cfg.data.size[0]), int(cfg.data.size[1]))),
            transforms.ToTensor()]
        )

        # load all images in folder
        img_path = str(Path(in_path, 'render.png'))

        # read first image in dir, change if required
        image = tfm(io.read_image(img_path))[None]
        image = image.to(device)

        return image


    base_image = load_image(Path(args.input_path, 'out'))

    # load model with weights
    def load_state_dict(weights):
        weights_path = str(Path(weights))
        state_dict = torch.load(weights_path)

        return state_dict

    
    model = NeuralMaterial(cfg.model)
    model.eval()
    model.to(device)


    state_dict_pre = load_state_dict(Path(args.model, 'checkpoint', 'latest.ckpt'))
    state_dict_base = load_state_dict(args.weight_path)



    model.load_state_dict(state_dict_pre)

    z1, _, _, _ = model.encode(base_image, 'test')

    output_path = Path(args.input_path, 'interps')
    output_path.mkdir(parents=True, exist_ok=True)

    directions = glob("./data/*/")
    sem_id = 0
    for mat_dir in directions:
        sTime = time.time()
        print(f"Generating semantic {sem_id+1}/{len(directions)}")
        
        model.load_state_dict(state_dict_pre)

        state_dict_sem = load_state_dict(Path(mat_dir, 'weights.ckpt'))
        sem_img = load_image(mat_dir)
        z2, _, _, _ = model.encode(sem_img, 'test')

        dists = [0.2, -0.2]

        # sample noise
        h_res = round(args.h / 16) * 16
        w_res = round(args.w / 16) * 16
        x = torch.rand(1, cfg.model.w, h_res, w_res, device=device)

        sTime = time.time()
        for inter_idx in range(0, len(dists)):
            a = dists[inter_idx]
            z_inter = (1 - a) * z1 + a * z2
            
            state_dict_inter = {}

            for k in state_dict_pre.keys():            
                state_dict_inter[k] = (1 - a) * state_dict_base[k] + a * state_dict_sem[k]

            model.load_state_dict(state_dict_inter)

            # convert noise to brdf maps using CNN
            brdf_maps = model.decode(z_inter, x)

            # render brdf maps using differentiable rendering
            image_out = model.renderer(brdf_maps)

            # write weights to disk
            torch.save(state_dict_inter, str(Path(output_path, f'{sem_id}_{inter_idx+1}_weights.ckpt')))

            # write outputs to disk
            save_png(image_out, str(Path(output_path, f'{sem_id}_{inter_idx+1}_render.png')), gamma=1.0)

            for k, v in brdf_maps.items():

                if k == 'normal':
                    v = (v + 1) / 2
                
                save_png(v, str(Path(output_path, f'{sem_id}_{inter_idx+1}_{k}.png')), gamma=2.2)
        print(f"Generating all interps for one semantic {time.time()-sTime:.2f}")
        sem_id += 1