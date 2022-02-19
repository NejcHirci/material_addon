import os
import glob
import sys
import numpy as np
sys.path.insert(1, 'materialGAN/src/')
from optim import loadLightAndCamera
from render import *

def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list

def renderTexFromTex(textures, tex_res, res, size, lp, cp, L, fn_im):
    if res > tex_res:
        print("[Warning in render.py::renderTex()]: request resolution is larger than texture resolution")
        exit()
    renderObj = Microfacet(res=tex_res, size=size)
    light = th.from_numpy(L).cuda() if th.cuda.is_available() else th.from_numpy(L)
    im = renderObj.eval(textures, lightPos=lp, \
        cameraPos=cp, light=light)
    im = gyApplyGamma(gyTensor2Array(im[0,:].permute(1,2,0)), 1/2.2)
    im = gyArray2PIL(im)
    if res < tex_res:
        im = im.resize((res, res), Image.LANCZOS)
    if fn_im is not None:
        im.save(fn_im)
    return im

def renderAndSave(tex, res, size, lp, cp, li, num_render, save_dir):
    for i in range(num_render):
        fn_this = save_dir + '/%02d.png' % i
        render_this = renderTexFromTex(tex, res, res, size, lp[i,:], cp[i,:], li, fn_im=fn_this)



root_dir = 'materialGAN/data/'
in_dir = 'D:\\materials\\NeuralMaterial\\'
out_dir = 'D:\\materials\\MaterialGAN\\'


mat_list = gyListNames(in_dir + '*')

# load light and camera position
light_pos, camera_pos, im_size, light = loadLightAndCamera(root_dir)

for id, mat in enumerate(mat_list):

    mat_in_dir = os.path.join(in_dir, mat)
    mat_in_dir = os.path.join(mat_in_dir, 'out')
    mat_out_dir = os.path.join(out_dir, mat)

    if os.path.isfile(os.path.join(mat_in_dir, 'tex.jpg')):
        textures, res = png2tex(os.path.join(mat_in_dir, 'tex.jpg'))
    else:
        textures, res = pngs2tex(mat_in_dir)

    # save initial texture and rendering
    renderAndSave(textures, res, im_size, light_pos, camera_pos, light, 9,
            mat_out_dir)

    np.savetxt(os.path.join(mat_out_dir, 'camera_pos.txt'), camera_pos, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(mat_out_dir, 'image_size.txt'), np.array([im_size]), delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(mat_out_dir, 'light_power.txt'), np.array([1500,1500,1500]).reshape(1,3), delimiter=',', fmt='%.4f')

    print("Rendered " + mat)
    
