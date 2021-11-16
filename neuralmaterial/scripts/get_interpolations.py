import argparse
from omegaconf import OmegaConf
import torch
from torch.serialization import load
import torchvision.io as io
import torchvision.transforms as transforms
from pathlib import Path
from glob import glob
import sys
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
    def load_model(weights):
        model = NeuralMaterial(cfg.model)
        weights_path = str(Path(weights))
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        return model

    base_model = load_model(args.weight_path)
    base_weights = base_model.decoder.state_dict()
    z1, _, _, _ = base_model.encode(base_image, 'test')

    output_path = Path(args.input_path, 'interps')
    output_path.mkdir(parents=True, exist_ok=True)

    directions = glob("./data/*/")
    sem_id = 0
    for mat_dir in directions:
        print(f"Generating semantic {sem_id+1}/{len(directions)}")
        inter_model = load_model(str(Path(mat_dir, 'weights.ckpt')))
        inter_weights = inter_model.decoder.state_dict()
        inter_image = load_image(str(Path(mat_dir)))

        z2, _, _, _ = inter_model.encode(inter_image, 'test')

        dists = [0.4, -0.4]

        # sample noise
        x = torch.rand(1, cfg.model.w, args.h, args.w, device=device)

        for inter_idx in range(0, len(dists)):
            a = dists[inter_idx]
            z_inter = (1 - a) * z1 + a * z2
            
            state_dict_inter = {}

            for k in base_weights.keys():            
                state_dict_inter[k] = (1 - a) * base_weights[k] + a * inter_weights[k]

            base_model.decoder.load_state_dict(state_dict_inter)

            # convert noise to brdf maps using CNN
            brdf_maps = base_model.decode(z_inter, x)

            # render brdf maps using differentiable rendering
            image_out = base_model.renderer(brdf_maps)

            # write weights to disk
            torch.save(base_model.state_dict(), str(Path(output_path, f'{sem_id}_{inter_idx+1}_weights.ckpt')))

            # write outputs to disk
            save_png(image_out, str(Path(output_path, f'{sem_id}_{inter_idx+1}_render.png')))

            for k, v in brdf_maps.items():

                if k == 'normal':
                    v = (v + 1) / 2
                
                save_png(v, str(Path(output_path, f'{sem_id}_{inter_idx+1}_{k}.png')))
        
        sem_id += 1