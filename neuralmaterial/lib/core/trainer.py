from tqdm import tqdm
from .logger import DummyLogger
import torch
from pathlib import Path
import torchvision.io as io

def save_png(img, path, gamma=1/2.2):
    img = img[0,:]
    if gamma < 1: img = img.clip(min=1e-6)
    img = img**gamma
    io.write_png((img * 255).byte().cpu(), path, compression_level=0)

class Trainer():
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = DummyLogger() if logger is None else logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.ckpt_dir = Path('checkpoint')
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        
        self.epoch = 1

    def _transfer_batch_to_gpu(self, batch):

        if torch.is_tensor(batch):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            res = {}
            for k, v in batch.items():
                res[k] = self._transfer_batch_to_gpu(v)
            return res
        elif isinstance(batch, list):
            res = []
            for v in batch:
                res.append(self._transfer_batch_to_gpu(v))
            return res
        else:
            raise TypeError("Invalid type for move_to")

    def _create_prog_bar(self):
        self.prog_bar = tqdm(desc='Train', total=0)

    def _update_prog_bar(self, mode, loss, step):
        if step % self.cfg.progress_bar_update == 0:
            self.prog_bar.set_postfix({f'{mode} Loss' : loss})
            self.prog_bar.update(self.cfg.progress_bar_update)
    
    def _reset_prog_bar(self, mode, total):
        self.prog_bar.reset(total=total)
        self.prog_bar.set_description(f'Epoch: {self.epoch} | {mode} step')
    
    def _save_checkpoint(self, state_dict_model):
        if self.epoch % self.cfg.save_checkpoint_every == 0:
            torch.save(state_dict_model, str(Path(self.ckpt_dir, 'latest.ckpt')))
            state_dict_metadata = {'epoch' : self.epoch}
            torch.save(state_dict_metadata, str(Path(self.ckpt_dir, 'metadata.pkl')))

    def _load_checkpoint(self, model):
        ckpt_path = Path(self.ckpt_dir, 'latest.ckpt')
        metadata_path = Path(self.ckpt_dir, 'metadata.pkl')

        if ckpt_path.is_file():
            state_dict = torch.load(str(ckpt_path))
            model.load_state_dict(state_dict)
            print('[INFO] Checkpoint loaded.')

        if metadata_path.is_file():
            metadata = torch.load(str(metadata_path))
            self.epoch = metadata['epoch'] + 1

            print('[INFO] Metadata loaded.')

    def _print_dataset_size(self, dataset, mode):
        print(f'[Data] {dataset.__len__()} {mode} samples')

    def finetune(self, model, image, steps, out_path=None):

        mode = 'finetune'

        model.finetuning_start()
        model.register_device(self.device)
        model.train()
        
        for step in range(1, steps+1):
            image = self._transfer_batch_to_gpu(image)
            outputs = model.forward_step(image, 'test')
            loss = outputs['metrics']['loss']
            model.after_train_step()
            model.backprop(loss)

            if step % 100 == 0:
                print(f'Epoch: [{step}/{steps}]   Loss: {loss.item():.2f}')

                if out_path is not None:
                    torch.save(model.state_dict(), Path(out_path, 'weights.ckpt'))
                    render = outputs['images']['image_out']
                    save_png(render, str(Path(out_path, 'render.png')))
                    albedo = outputs['images']['albedo']
                    save_png(albedo, str(Path(out_path,'albedo.png')))
                    specular = outputs['images']['specular']
                    save_png(specular, str(Path(out_path,'specular.png')))
                    rough = outputs['images']['rough']
                    save_png(rough, str(Path(out_path,'rough.png')))
                    normal = (outputs['images']['normal'] + 1) / 2
                    save_png(normal, str(Path(out_path,'normal.png')))               

        return model

    def fit(self, model, data):

        data.setup()

        model.training_start()
        model.register_device(self.device)

        train_dl = data.train_dataloader()
        val_dl = data.val_dataloader()

        n_train = len(train_dl)
        n_val = len(val_dl)

        self._print_dataset_size(train_dl, 'train')
        self._print_dataset_size(train_dl, 'val')

        self._load_checkpoint(model)

        if self.cfg.print_num_params:
            model.print_num_params()

        self._create_prog_bar()

        while True:
            
            model.train()
            mode = 'train'
            self._reset_prog_bar(mode, n_train)

            for step, train_batch in enumerate(train_dl, start=1):
                train_batch = self._transfer_batch_to_gpu(train_batch)
                outputs = model.forward_step(train_batch, mode)
                loss = outputs['metrics']['loss']
                model.after_train_step()
                model.backprop(loss)
                
                self.logger.update_metrics(outputs['metrics'], mode)
                self._update_prog_bar(mode, loss.item(), step)

            self.logger.log_dict(outputs, mode, self.epoch, step)
            self.logger.reset_metrics()
            self._save_checkpoint(model.state_dict())

            # run validation loop if conditions apply
            if self.epoch % self.cfg.val_every == 0:
                
                model.eval()
                mode = 'val'
                self._reset_prog_bar(mode, n_val)

                with torch.no_grad():
                    for step, val_batch in enumerate(val_dl, start=1):
                        val_batch = self._transfer_batch_to_gpu(val_batch)
                        outputs = model.forward_step(val_batch, mode)
                        loss = outputs['metrics']['loss']
                        self.logger.update_metrics(outputs['metrics'], mode)
                        self._update_prog_bar(mode, loss.item(), step)

                self.logger.log_dict(outputs, mode, self.epoch, step)
                self.logger.reset_metrics()

            if self.epoch == self.cfg.epochs:
                model.training_end()
                break
            
            self.epoch += 1            
