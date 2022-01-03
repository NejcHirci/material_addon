import argparse
from omegaconf import OmegaConf
import torch
import torchvision.io as io
import torchvision.transforms as transforms
from pathlib import Path
import sys
sys.path.insert(0, './')

from lib.core.utils import seed_everything
from lib.core.trainer import Trainer
from lib.main import NeuralMaterial

def save_png(img, path, gamma=1):
    img = img[0,:]
    if gamma < 1: img = img.clip(min=1e-6)
    img = img**gamma
    io.write_png((img * 255).byte().cpu(), path, compression_level=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve BRDF decomposition from stationary flash image.')
    parser.add_argument('--model', type=str, required=True,
                        help='yaml config file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='path to input images')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to output folder')
    parser.add_argument('--epochs', type=int, required=False, default=3000,
                        help='number of epochs')
    parser.add_argument('--gpu', type=bool, required=False, default=False,
                        help='use GPU / CPU')
    parser.add_argument('--h', type=int, required=False, default=384,
                        help='output height')
    parser.add_argument('--w', type=int, required=False, default=512,
                        help='output width')
    parser.add_argument('--reseed', dest='reseed', action='store_true')
    parser.set_defaults(reseed=False)
    parser.add_argument('--seed', type=int, required=False, default=42)

    args = parser.parse_args()
    finetuning_steps = args.epochs

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')

    # load config
    cfg = OmegaConf.load(str(Path(args.model, '.hydra', 'config.yaml')))
    seed_everything(args.seed)

    # load image
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((int(cfg.data.size[0]), int(cfg.data.size[1]))),
        transforms.ToTensor()]
    )

    # load model with weights
    model = NeuralMaterial(cfg.model)

    weights_path = str(Path(args.model, 'checkpoint', 'latest.ckpt'))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = Path(output_path, 'weights.ckpt')

    # load all images in folder
    image_dirs = [str(p) for p in Path(args.input_path).iterdir() if p.is_file() and (p.suffix == '.png' or p.suffix == '.jpg')]

    # read first image in dir, change if required
    image = tfm(io.read_image(image_dirs[0]))[None]
    image = image.to(device)

    if args.reseed:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        if weights_path.is_file():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)

        trainer = Trainer(cfg.trainer)
        model = trainer.finetune(model, image, finetuning_steps, out_path=Path(output_path))
        torch.save(model.state_dict(), weights_path)

    model.eval()
    model.to(device)

    print("Generating output")

    # run forward pass and retrieve brdf decomposition
    image_out, brdf_maps, _, _ , _ = model.forward(image, 'test', size=(args.h, args.w))

    # write outputs to disk
    save_png(image_out, str(Path(output_path,'render.png')))

    for k, v in brdf_maps.items():

        if k == 'normal':
            v = (v + 1) / 2
        
        save_png(v, str(Path(output_path, f'{k}.png')))

    
