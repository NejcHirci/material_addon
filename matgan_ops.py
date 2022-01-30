# The following script contains classes with necessary Blender operators
# for the MaterialGAN approach in material generation and editing.
#
# Code from MaterialGAN paper is stored in materialGAN directory
# and is available on the following repository: https://github.com/tflsguoyu/materialgan
#
# In adition I added a superresolution upscaling step to the approach, which uses
# code from the following paper https://github.com/yinboc/liif.

from email.mime import base
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

import bpy
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

base_script_path = Path(__file__).parent.resolve()

PYTHON_EXE = os.path.join(str(base_script_path), 'venv\\Scripts\\python.exe')

def check_remove_img(name):
    if name in bpy.data.images:
        image = bpy.data.images[name]
        bpy.data.images.remove(image)

def replace_file(src_path, dst_path, retries=10, sleep=0.1):
    for i in range(retries):
        try:
            os.replace(src_path, dst_path)
        except WindowsError:
            time.sleep(sleep)
        else:
            break
    
# Function for updating textures during material generation.
def update_matgan(base_path):
    active_obj = bpy.context.scene.objects.active

    if active_obj:
        base_name = active_obj.name
        if base_name not in bpy.data.materials:
            mat = bpy.data.materials["matgan_mat"].copy()
            mat.name = base_name
        else:
            mat = bpy.data.materials[base_name]
    else:
        base_name = "base"
        mat = bpy.data.materials["matgan_mat"]
    
    nodes = mat.node_tree.nodes

    albedo = nodes.get("Image Texture")
    specular = nodes.get("Image Texture.001")
    rough = nodes.get("Image Texture.002")
    normal = nodes.get("Image Texture.003")

    if os.path.isfile(os.path.join(base_path, 'albedo.png')):
        check_remove_img(f'{base_name}-matgan-render.png')
        img = bpy.data.images.load(os.path.join(base_path, 'render.png'))
        img.name = f'{base_name}-matgan-render.png'
        check_remove_img(f'{base_name}-matgan-albedo.png')
        img = bpy.data.images.load(os.path.join(base_path, 'albedo.png'))
        img.name = f'{base_name}-matgan-albedo.png'
        albedo.image = img
        check_remove_img(f'{base_name}-matgan-specular.png')
        img = bpy.data.images.load(os.path.join(base_path, 'specular.png'))
        img.name = f'{base_name}-matgan-specular.png'
        specular.image = img
        check_remove_img(f'{base_name}-matgan-rough.png')
        img = bpy.data.images.load(os.path.join(base_path, 'rough.png'))
        img.name = f'{base_name}-matgan-rough.png'
        rough.image = img
        check_remove_img(f'{base_name}-matgan-normal.png')
        img = bpy.data.images.load(os.path.join(base_path, 'normal.png'))
        img.name = f'{base_name}-matgan-normal.png'
        normal.image = img

class MAT_OT_MATGAN_Generator(Operator):
    bl_idname = "matgan.mat_from_images"
    bl_label = "Generate material"
    bl_description = "Generate base material from flash images."

    _popen = None
    
    @classmethod
    def poll(self, context):
        return not ("Epoch" in bpy.context.scene.matgan_properties.progress) and not ("Format" in bpy.context.scene.matgan_properties.progress) \
                and os.path.isdir(os.path.join(bpy.context.scene.matgan_properties.directory, 'input'))


    def execute(self, context):
        gan = bpy.context.scene.matgan_properties
        base_dir = os.path.join(gan.directory, 'input')
        
        in_dir  = base_dir
        out_dir = os.path.join(gan.directory, 'out')

        cp_dir  = './materialGAN/data/pretrain/'
        vgg_dir = cp_dir + 'vgg_conv.pt'

        N = gan.num_rend
        epochs = gan.epochs
        epochW = 10
        epochN = 10
        loss = [1000, 0.001, -1, -1]
        lr = 0.02

        # Call to generate texture maps
        process = subprocess.Popen([PYTHON_EXE, '-u', './materialGAN/src/optim.py', 
                '--in_dir', in_dir,
                '--out_dir', out_dir,
                '--vgg_weight_dir', vgg_dir,
                '--num_render_used', str(N),
                '--epochs', str(epochs),
                '--sub_epochs', str(epochW), str(epochN),
                '--loss_weight', str(loss[0]), str(loss[1]), str(loss[2]), str(loss[3]),
                '--optim_latent', 
                '--lr', str(lr),
                '--gan_latent_init',  cp_dir + 'latent_avg_W+_256.pt',
                '--gan_noise_init', cp_dir + 'latent_const_N_256.pt'], stdout=subprocess.PIPE, cwd=str(base_script_path))

        MAT_OT_MATGAN_Generator._popen = process

        gan.progress = 'Epoch: [{}/{}]   Loss: 0.0 0.0'.format(1, gan.epochs)
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}


class MAT_OT_MATGAN_GetInterpolations(Operator):
    bl_idname = "matgan.get_interpolations"
    bl_label = "Get interpolations"
    bl_description = "Generate interpolations in discovered directions"

    _popen = None

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.matgan_properties.progress

    def execute(self, context):
        gan = bpy.context.scene.matgan_properties
        save_dir = os.path.join(gan.directory, 'interps')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out = os.path.join(gan.directory, 'out')

        latent_path = os.path.join(out, 'optim_latent.pt')
        noise_path = os.path.join(out, 'optim_noise.pt')

        # Call to generate texture maps
        process = subprocess.Popen([PYTHON_EXE, '-u', './sefa/get_directions.py', 
                '--save_dir', save_dir,
                '--latent_path', latent_path,
                '--noise_path', noise_path
                ], stdout=subprocess.PIPE, cwd=str(base_script_path))
        MAT_OT_MATGAN_GetInterpolations._popen = process

        gan.progress = 'Generating interpolations in directions'
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}


class MAT_OT_MATGAN_StopGenerator(Operator):
    bl_idname = "matgan.stop_generator"
    bl_label = "Stop generator material"
    bl_description = "Stop generate base material from flash images."

    @classmethod
    def poll(self, context):
        return MAT_OT_MATGAN_Generator._popen
    
    def execute(self, context):
        MAT_OT_MATGAN_Generator._popen.terminate()
        return {'FINISHED'}


class MAT_OT_MATGAN_InputFromFlashImage(Operator):
    bl_idname = "matgan.input_from_images"
    bl_label = "Format flash images"
    bl_description = "Generate input images from flash images for material generation."

    _popen = None

    @classmethod
    def poll(self, context):
        return "Ready to format." in bpy.context.scene.matgan_properties.progress

    def execute(self, context):
        base_dir = bpy.context.scene.matgan_properties.directory
        in_dir  = base_dir
        out_dir = os.path.join(base_dir, 'input')

        print(PYTHON_EXE)
        print(base_script_path)

        process = subprocess.Popen([PYTHON_EXE, '-u', './materialGAN/tools/generate_inputs.py',
                    '--in_dir', in_dir,
                    '--out_dir', out_dir], stdout=subprocess.PIPE, cwd=str(base_script_path))

        MAT_OT_MATGAN_InputFromFlashImage._popen = process

        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}


class MAT_OT_MATGAN_EditMove(Operator):
    bl_idname = "matgan.edit_move"
    bl_label = "Move material in desired latent direction."
    bl_description = "Finds 9 neighbouring materials in latent space directions from the newly chosen one."

    direction : bpy.props.StringProperty(default="")

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.matgan_properties.progress

    def execute(self, context):
        gan = bpy.context.scene.matgan_properties
        interp_dir = os.path.join(gan.directory, 'interps')

        # First unlink files
        check_remove_img('matgan-render.png')
        check_remove_img('matgan-albedo.png')
        check_remove_img('matgan-rough.png')
        check_remove_img('matgan-specular.png')
        check_remove_img('matgan-normal.png')

        new_latent_path = os.path.join(interp_dir, f'{self.direction}_1_optim_latent.pt')
        new_noise_path = os.path.join(interp_dir, f'{self.direction}_1_optim_noise.pt')
        new_render_path = os.path.join(interp_dir, f'{self.direction}_1_render.png')
        new_albedo_path = os.path.join(interp_dir, f'{self.direction}_1_albedo.png')
        new_rough_path = os.path.join(interp_dir, f'{self.direction}_1_rough.png')
        new_specular_path = os.path.join(interp_dir, f'{self.direction}_1_specular.png')
        new_normal_path = os.path.join(interp_dir, f'{self.direction}_1_normal.png')

        # Rename old files
        out = os.path.join(gan.directory, 'out')
        old_latent_path = os.path.join(out, 'optim_latent.pt')
        replace_file(old_latent_path, os.path.join(out, 'old_optim_latent.pt'))
        old_noise_path = os.path.join(out, 'optim_noise.pt')
        replace_file(old_noise_path, os.path.join(out, 'old_optim_noise.pt'))
        old_render_path = os.path.join(out, 'render.png')
        replace_file(old_render_path, os.path.join(out, 'old_render.png'))
        old_albedo_path = os.path.join(out, 'albedo.png')
        replace_file(old_albedo_path, os.path.join(out, 'old_albedo.png'))
        old_rough_path = os.path.join(out, 'rough.png')
        replace_file(old_rough_path, os.path.join(out, 'old_rough.png'))
        old_specular_path = os.path.join(out, 'specular.png')
        replace_file(old_specular_path, os.path.join(out, 'old_specular.png'))
        old_normal_path = os.path.join(out, 'normal.png')
        replace_file(old_normal_path, os.path.join(out, 'old_normal.png'))

        # Copy and replace old files
        shutil.move(new_latent_path, old_latent_path)
        shutil.move(new_noise_path, old_noise_path)
        shutil.move(new_render_path, old_render_path)
        shutil.move(new_albedo_path, old_albedo_path)
        shutil.move(new_rough_path, old_rough_path)
        shutil.move(new_specular_path, old_specular_path)
        shutil.move(new_normal_path, old_normal_path)

        # Update material textures
        update_matgan(out)

        # Call to generate texture maps
        process = subprocess.Popen([PYTHON_EXE, '-u', './sefa/get_directions.py', 
                '--save_dir', interp_dir,
                '--latent_path', old_latent_path,
                '--noise_path', os.path.join(out, 'optim_noise.pt')
                ], stdout=subprocess.PIPE, cwd=str(base_script_path))
        MAT_OT_MATGAN_GetInterpolations._popen = process

        gan.progress = 'Generating interpolations in directions'
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}


class MAT_OT_MATGAN_SuperResolution(Operator):
    bl_idname = "matgan.super_res"
    bl_label = "Upscale original textures."
    bl_description = "Converts textures to LIIF and upscales the image"
    
    _popen = None

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.matgan_properties.progress

    def execute(self, context):
        gan = bpy.context.scene.matgan_properties
        base_path = gan.directory
        in_path = os.path.join(base_path, 'out')
        model_path = './liif/pretrain/edsr-baseline-liif.pth'

        h = gan.h_res
        w = gan.w_res

        process = subprocess.Popen([PYTHON_EXE, './liif/demo.py',
            '--dir', in_path,
            '--model', model_path,
            '--resolution', "{},{}".format(h, w),
            '--gpu', "0"], stdout=subprocess.PIPE, cwd=str(base_script_path))
        
        print(process.args)

        gan.progress = "Upscaling material"
        MAT_OT_MATGAN_SuperResolution._popen = process
        print(process)
        bpy.ops.wm.modal_status_updater()
        return {'FINISHED'}


class MAT_OT_MATGAN_FileBrowser(Operator, ImportHelper):
    """File browser operator"""
    bl_idname= "matgan.file_browser"
    bl_label = "Selects folder with data"
    
    filename_ext = ""

    def execute(self, context):
        fdir = self.properties.filepath
        bpy.context.scene.matgan_properties.directory = os.path.dirname(fdir)
        fdir = os.path.dirname(fdir)
        if os.path.isdir(os.path.join(fdir, 'out')):
            bpy.context.scene.matgan_properties.progress = "Material found."
            update_matgan(os.path.join(fdir, 'out'))        
        else:
            bpy.context.scene.matgan_properties.progress = "Ready to format."
        return {'FINISHED'}
