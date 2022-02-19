import subprocess
import global_var
from util import *
from render import *
from loss import *
from torch.optim import Adam
import shutil
from applications import interpolate_single_texture
sys.path.insert(1, 'materialGAN/higan/models/')
from stylegan2_generator import StyleGAN2Generator

np.set_printoptions(precision=4, suppress=True)

# th.autograd.set_detect_anomaly(True)

def save_args(args, dir):
    with open(dir, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def loadLightAndCamera(in_dir):
    camera_pos = np.loadtxt(os.path.join(in_dir, 'camera_pos.txt'), delimiter=',').astype(np.float32)

    light_pos = np.loadtxt(os.path.join(in_dir, 'light_pos.txt'), delimiter=',').astype(np.float32)

    im_size = np.loadtxt(os.path.join(in_dir, 'image_size.txt'), delimiter=',')
    im_size = float(im_size)
    light = np.loadtxt(os.path.join(in_dir, 'light_power.txt'), delimiter=',')

    return light_pos, camera_pos, im_size, light


def loadTarget(in_dir, res, num_render):

    def loadTargetToTensor(dir, res):
        target = Image.open(dir)
        if not target.width == res:
            target = target.resize((res, res), Image.LANCZOS)
        target = gyPIL2Array(target)
        target = th.from_numpy(target).permute(2,0,1)
        return target

    rendered = th.zeros(num_render, 3, res, res)
    for i in range(num_render):
        rendered[i,:] = loadTargetToTensor(os.path.join(in_dir,'%02d.png' % i), res)
    if args.gpu > -1:
        rendered = rendered.cuda()

    texture_fn = os.path.join(in_dir, 'tex.png')
    if os.path.exists(texture_fn):
        textures, res0 = png2tex(texture_fn)
    else:
        textures = None

    return rendered, textures

def initTexture(init_from, res):
    if init_from == 'random':
        textures_tmp = th.rand(1,9,res,res)
        textures = textures_tmp.clone()
        textures[:,0:5,:,:] = textures_tmp[:,0:5,:,:] * 2 - 1
        textures[:,5,:,:] = textures_tmp[:,5,:,:] * 1.3 - 0.3
        textures[:,6:9,:,:] = textures_tmp[:,6:9,:,:] * 2 - 1
    else:
        textures, _ = png2tex(init_from)
        if res != textures.shape[-1]:
            print('The loaded initial texture has a wrong resolution!')
            exit()
    return textures

def initLatent(genObj, type, init_from):
    if init_from == 'random':
        if type == 'z':
            latent = th.randn(1,512).cuda() if args.gpu > -1 else th.randn(1,512)
        elif type == 'w':
            latent = th.randn(1,512).cuda() if args.gpu > -1 else th.randn(1,512)
            latent = genObj.net.mapping(latent)
        elif type == 'w+':
            latent = th.randn(1,512).cuda() if args.gpu > -1 else th.randn(1,512)
            latent = genObj.net.mapping(latent)
            latent = genObj.net.truncation(latent)
        else:
            print('--gan_latent_type should be z|w|w+')
            exit()
    else:
        if os.path.exists(init_from):
            latent = th.load(init_from).cuda() if args.gpu > -1 else th.load(init_from)
        else:
            print('Can not find latent vector ', init_from)
            exit()

    return latent

def updateTextureFromLatent(genObj, type, latent):
    if type == 'z':
        latent = genObj.net.mapping(latent)
        latent = genObj.net.truncation(latent)
    elif type == 'w':
        latent = genObj.net.truncation(latent)
    elif type == 'w+':
        pass

    textures = genObj.net.synthesis(latent)
    textures_tmp = textures.clone()
    textures_tmp[:,0:5,:,:] = textures[:,0:5,:,:].clamp(-1,1)
    textures_tmp[:,5,:,:] = textures[:,5,:,:].clamp(-0.3,1)
    textures_tmp[:,6:9,:,:] = textures[:,6:9,:,:].clamp(-1,1)

    return textures_tmp

def renderAndSave(tex, res, size, lp, cp, li, num_render, save_dir, tmp_dir, epoch):
    fn = os.path.join(save_dir,'tex.png')
    fn2 = os.path.join(save_dir,'rendered.png')
    png = tex2png(tex, fn)
    
    tex2pngs(tex, save_dir)

    render_all = None
    for i in range(num_render):
        fn_this = save_dir + '/%02d.png' % i
        render_this = renderTex(fn, 256, size, lp[i,:], cp[i,:], li, fn_im=fn_this)
        if i == 0:
            render_this.save(save_dir + '/render.png')
        render_all = gyConcatPIL_h(render_all, render_this)
        png = gyConcatPIL_h(png, render_this)

    render_all.save(fn2)

    # gyCreateThumbnail(fn2, w=128*num_render, h=128)
    png.save(os.path.join(tmp_dir, 'epoch_%05d.jpg' % epoch))

def optim(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.manual_seed(args.seed)

    now = datetime.now();

    if args.gpu > -1:
        th.cuda.set_device(args.gpu)

    mat = args.mat_fn
    res = args.im_res
    in_this_dir = os.path.join(args.in_dir, mat)
    out_this_dir = os.path.join(args.out_dir, mat)
    gyCreateFolder(out_this_dir)
    out_this_tmp_dir = os.path.join(out_this_dir, 'tmp')
    gyCreateFolder(out_this_tmp_dir)

    save_args(args, os.path.join(out_this_dir, 'args.txt'))

    light_pos, camera_pos, im_size, light = loadLightAndCamera(in_this_dir)

    rendered_ref, textures_ref = loadTarget(in_this_dir, res, 9)



    # initial textures
    if args.optim_latent:
        genObj = StyleGAN2Generator('svbrdf')
        # initialize noise space
        global_var.init_global_noise(res, args.gan_noise_init)

        # initialize latent space
        latent = initLatent(genObj, args.gan_latent_type, args.gan_latent_init)
        latent = Variable(latent, requires_grad=True)

        # GAN generation
        texture_pre = updateTextureFromLatent(genObj, args.gan_latent_type, latent)

    else:
        texture_pre = initTexture(args.tex_init, res)
        texture_pre = Variable(texture_pre, requires_grad=True)

    texture_init = texture_pre.clone().detach()


    # save initial texture and rendering
    renderAndSave(texture_pre, res, im_size, light_pos, camera_pos, light, args.num_render,
            out_this_dir, out_this_tmp_dir, 0)

    # initial loss
    loss_list_all = []
    loss_fn =  os.path.join(out_this_dir, 'loss.txt')
    lossObj = Losses(args, texture_init, textures_ref, rendered_ref, res, im_size, light_pos, camera_pos)

    if not args.optim_latent:
        optimizer = Adam([texture_pre], lr=args.lr, betas=(0.9, 0.999))
        optim_strategy_this = 'LN'

    # optimization
    for epoch in range(args.epochs):
        # update what to optimize
        if args.optim_latent:
            if args.optim_strategy == 'L+N':
                optimizer = Adam([latent] + global_var.noises, lr=args.lr, betas=(0.9, 0.999))
                optim_strategy_this = 'LN'
                # print('@@@ optim both @@@')
            elif args.optim_strategy == 'L':
                optimizer = Adam([latent], lr=args.lr, betas=(0.9, 0.999))
                optim_strategy_this = 'L'
                # print('@@@ optim latent @@@')
            elif args.optim_strategy == 'N':
                optimizer = Adam(global_var.noises, lr=args.lr, betas=(0.9, 0.999))
                optim_strategy_this = 'N'
                # print('@@@ optim noise @@@')
            else:
                epoch_tmp = epoch % (args.sub_epochs[0]+args.sub_epochs[1])
                if int(epoch_tmp / args.sub_epochs[0]) == 0:
                    optimizer = Adam([latent], lr=args.lr, betas=(0.9, 0.999))
                    optim_strategy_this = 'L'
                    # print('@@@ optim latent @@@')
                else:
                    optimizer = Adam(global_var.noises, lr=args.lr, betas=(0.9, 0.999))
                    optim_strategy_this = 'N'
                    # print('@@@ optim noise @@@')

        # compute loss
        loss, loss_list = lossObj.eval(texture_pre, light, optim_strategy_this, epoch)
        loss_list_all.append(loss_list)
        np.savetxt(loss_fn, np.vstack(loss_list_all), fmt='%.4f', delimiter=',')

        # update latent/textures
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # undate textures
        if args.optim_latent:
            texture_pre = updateTextureFromLatent(genObj, args.gan_latent_type, latent)

        # save output
        if (epoch+1) % 100 == 0 or epoch == 0:
            print('Epoch: [%d/%d]   Loss: %.2f %.2f' % (epoch+1, args.epochs, loss_list[0], loss_list[1]), flush=True)
            renderAndSave(texture_pre, res, im_size, light_pos, camera_pos, light, args.num_render,
                out_this_dir, out_this_tmp_dir, epoch+1)
            plotAndSave(np.vstack(loss_list_all), os.path.join(out_this_tmp_dir, 'loss.png'))
            if args.optim_latent:
                th.save(latent, os.path.join(out_this_dir, 'optim_latent.pt'))
                th.save(global_var.noises, os.path.join(out_this_dir, 'optim_noise.pt'))

    shutil.rmtree(out_this_tmp_dir)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Optimization -- GY')
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--mat_fn', type=str, default='')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--vgg_weight_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--im_res', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim_latent', action='store_true')
    parser.add_argument('--tex_init', type=str, default='random')
    parser.add_argument('--num_render', type=int, default=9)
    parser.add_argument('--num_render_used', type=int, default=5)
    parser.add_argument('--gan_latent_type', type=str, default='w+')
    parser.add_argument('--gan_latent_init', type=str, default='random')
    parser.add_argument('--gan_noise_init', type=str, default='random')
    parser.add_argument('--sub_epochs', type=int, nargs='+')
    parser.add_argument('--loss_weight', type=float, nargs='+')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--embed_tex', action='store_true')
    parser.add_argument('--jittering', action='store_true')

    args = parser.parse_args()

    # ours
    # args.vgg_layer_weight_w  = [0.125, 0.125, 0.125, 0.125]
    # args.vgg_layer_weight_n  = [0.125, 0.125, 0.125, 0.125]

    args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    args.vgg_layer_weight_wn = [0.125, 0.125, 0.125, 0.125]

    if args.sub_epochs[0] == 0:
        if args.sub_epochs[1] == 0:
            args.optim_strategy = 'L+N'
        else:
            args.optim_strategy = 'N'
    else:
        if args.sub_epochs[1] == 0:
            args.optim_strategy = 'L'
        else:
            args.optim_strategy = 'L|N'

    if args.seed:
        pass
    else:
        args.seed = random.randint(0, 2**31 - 1)

    args.epochs = max(args.sub_epochs[0] + args.sub_epochs[1], args.epochs)

    optim(args)
    