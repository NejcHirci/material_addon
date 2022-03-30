# The following script contains classes with necessary Blender operators
# for the Neural material approach in material generation and editing.
#
# Code from Neural Material paper is stored in neuralmaterial directory
# and is available on the following repository: https://github.com/henzler/neuralmaterial

import os
import subprocess
import sys
import shutil
from pathlib import Path
import time

import bpy
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

base_script_path = Path(__file__).parent.resolve()

PYTHON_EXE = os.path.join(str(base_script_path), 'venv\\Scripts\\python.exe')

def check_remove_img(name):
    if name in bpy.data.images:
        image = bpy.data.images[name]
        bpy.data.images.remove(image)

# Function for updating textures during material generation.
def update_neural(obj, base_path):
    if obj:
        base_name = f"{obj.name}_neural_mat"
        if base_name not in bpy.data.materials:
            mat = bpy.data.materials["neural_mat"].copy()
            mat.name = base_name
        else:
            mat = bpy.data.materials[base_name]
    else:
        base_name = "base_neural"
        mat = bpy.data.materials["neural_mat"]
        
    nodes = mat.node_tree.nodes

    albedo = nodes.get("Image Texture")
    specular = nodes.get("Image Texture.001")
    rough = nodes.get("Image Texture.002")
    normal = nodes.get("Image Texture.003")

    if os.path.isfile(os.path.join(base_path, 'albedo.png')):
        check_remove_img(f'{base_name}_render.png')
        img = bpy.data.images.load(os.path.join(base_path, 'render.png'))
        img.name = f'{base_name}_render.png'
        check_remove_img(f'{base_name}_albedo.png')
        img = bpy.data.images.load(os.path.join(base_path, 'albedo.png'))
        img.name = f'{base_name}_albedo.png'
        albedo.image = img
        check_remove_img(f'{base_name}_specular.png')
        img = bpy.data.images.load(os.path.join(base_path, 'specular.png'))
        img.name = f'{base_name}_specular.png'
        specular.image = img
        check_remove_img(f'{base_name}_rough.png')
        img = bpy.data.images.load(os.path.join(base_path, 'rough.png'))
        img.name = f'{base_name}_rough.png'
        rough.image = img
        check_remove_img(f'{base_name}_normal.png')
        img = bpy.data.images.load(os.path.join(base_path, 'normal.png'))
        img.name = f'{base_name}_normal.png'
        normal.image = img

def replace_file(src_path, dst_path, retries=10, sleep=0.1):
    for i in range(retries):
        try:
            os.replace(src_path, dst_path)
        except WindowsError:
            import pdb
            pdb.set_trace()
            time.sleep(sleep)
        else:
            break


class MAT_OT_NEURAL_GetInterpolations(Operator):
    bl_idname = "neural.get_interpolations"
    bl_label = "Get interpolations"
    bl_description = "Generate interpolations in discovered directions"

    _popen = None

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.neural_properties.progress

    def execute(self, context):
        neural = bpy.context.scene.neural_properties

        in_dir  = neural.directory
        weight_dir = os.path.join(neural.directory, 'out', 'weights.ckpt')

        model_path = './trainings/Neuralmaterial'

        max_w = min(neural.w_res, 1024)
        max_h = min(neural.h_res, 1024)

        # Call to generate texture maps
        process = subprocess.Popen([PYTHON_EXE, '-u', './scripts/get_interpolations.py',
                '--model', model_path,
                '--input_path', in_dir,
                '--weight_path', weight_dir,
                '--h', str(max_h),
                '--w', str(max_w)], stdout=subprocess.PIPE, cwd=str(Path(base_script_path, 'neuralmaterial')))

        MAT_OT_NEURAL_GetInterpolations._popen = process

        neural.progress = 'Generating interpolations ...'
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}

class MAT_OT_NEURAL_FileBrowser(Operator, ImportHelper):
    """File browser operator"""
    bl_idname= "neural.file_browser"
    bl_label = "Selects folder with data"
    
    filename_ext = ""

    def invoke(self, context, event):
        self.filepath = bpy.context.scene.neural_properties.directory
        wm = context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        fdir = self.properties.filepath
        gan = bpy.context.scene.neural_properties
        gan.directory = os.path.dirname(fdir)
        fdir = os.path.dirname(fdir)

        active_obj = bpy.context.active_object
        if active_obj:
            # Store base material path for later saving
            active_obj["Neural_Path"] = fdir 

        if os.path.isdir(os.path.join(fdir, 'out')):
            gan.progress = "Material found."
            update_neural(active_obj, os.path.join(fdir, 'out'))        
        else:
            gan.progress = "Ready to generate."
        return {'FINISHED'}

class MAT_OT_NEURAL_Generator(Operator):
    bl_idname = "neural.generator"
    bl_label = "Generate neural material"
    bl_description = "Generate base material from flash images"

    _popen = None

    @classmethod
    def poll(self, context):
        return MAT_OT_NEURAL_Generator._popen is None and MAT_OT_NEURAL_GetInterpolations._popen is None

    def execute(self, context):
        neural = bpy.context.scene.neural_properties

        in_dir  = neural.directory
        out_dir = os.path.join(neural.directory, 'out')

        model_path = './trainings/Neuralmaterial'
        N = neural.num_rend
        epochs = str(neural.epochs)

        # Call to generate texture maps
        process = subprocess.Popen([PYTHON_EXE, '-u', './scripts/test.py',
                '--model', model_path,
                '--input_path', in_dir,
                '--output_path', out_dir,
                '--epochs', epochs,
                '--h', str(neural.h_res),
                '--w', str(neural.w_res)], stdout=subprocess.PIPE, cwd=str(Path(base_script_path, 'neuralmaterial')))

        MAT_OT_NEURAL_Generator._popen = process

        neural.progress = 'Epoch: [{}/{}]   Loss: 0.0 0.0'.format(1, neural.epochs)
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}

class MAT_OT_NEURAL_Reseed(Operator):
    bl_idname = "neural.reseed"
    bl_label = "Neural material reseed"
    bl_description = "Generate new learned material with new seed"

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.neural_properties.progress

    def execute(self, context):
        neural = bpy.context.scene.neural_properties

        in_dir  = neural.directory
        out_dir = os.path.join(neural.directory, 'out')

        if not Path(out_dir, 'weights.ckpt').is_file():
            neural.progress = 'Material not generated or corrupted.'
            return {'FINISHED'}

        model_path = './trainings/Neuralmaterial'
        epochs = str(neural.epochs)

        # Call to generate texture maps
        process = subprocess.Popen([PYTHON_EXE, '-u', './scripts/test.py',
                '--model', model_path,
                '--input_path', in_dir,
                '--output_path', out_dir,
                '--epochs', epochs,
                '--h', str(neural.h_res),
                '--w', str(neural.w_res),
                '--seed', str(neural.seed),
                '--reseed'], stdout=subprocess.PIPE, cwd=str(Path(base_script_path, 'neuralmaterial')))

        MAT_OT_NEURAL_Generator._popen = process

        neural.progress = 'Reseeding material with {}'.format(neural.seed)
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}
        
class MAT_OT_NEURAL_EditMove(Operator):
    bl_idname = "neural.edit_move"
    bl_label = "Move material in desired material directions."
    bl_description = "Finds 9 neighbouring materials in latent space directions from the newly chosen one."

    direction: bpy.props.StringProperty(default="")

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.neural_properties.progress


    def preprocess(self, context):
        if bpy.context.active_object:
            name = f"{bpy.context.active_object.name}_neural"
        else:
            name = "neural"
        
        # First unlink files
        check_remove_img(f'{name}_render.png')
        check_remove_img(f'{name}_albedo.png')
        check_remove_img(f'{name}_rough.png')
        check_remove_img(f'{name}_specular.png')
        check_remove_img(f'{name}_normal.png')

    def execute(self, context):
        gan = bpy.context.scene.neural_properties
        interp_dir = os.path.join(gan.directory, 'interps')

        self.preprocess(context)

        new_weight_path = os.path.join(interp_dir, f'{self.direction}_1_weights.ckpt')
        new_render_path = os.path.join(interp_dir, f'{self.direction}_1_render.png')
        new_albedo_path = os.path.join(interp_dir, f'{self.direction}_1_albedo.png')
        new_rough_path = os.path.join(interp_dir, f'{self.direction}_1_rough.png')
        new_specular_path = os.path.join(interp_dir, f'{self.direction}_1_specular.png')
        new_normal_path = os.path.join(interp_dir, f'{self.direction}_1_normal.png')

        # Rename old files
        out = os.path.join(gan.directory, 'out')
        old_weight_path = os.path.join(out, 'weights.ckpt')
        replace_file(old_weight_path, os.path.join(out, 'old_weights.ckpt'))
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
        shutil.move(new_weight_path, old_weight_path)
        shutil.move(new_render_path, old_render_path)
        shutil.move(new_albedo_path, old_albedo_path)
        shutil.move(new_rough_path, old_rough_path)
        shutil.move(new_specular_path, old_specular_path)
        shutil.move(new_normal_path, old_normal_path)

        # Update material textures
        update_neural(bpy.context.active_object, out)

        in_dir  = os.path.join(gan.directory)
        weight_dir = os.path.join(gan.directory, 'out', 'weights.ckpt')

        model_path = './trainings/Neuralmaterial'

        max_h = min(gan.h_res, 1024)
        max_w = min(gan.w_res, 1024)

        # Call to generate texture maps
        process = subprocess.Popen([PYTHON_EXE, '-u', './scripts/get_interpolations.py',
                '--model', model_path,
                '--input_path', in_dir,
                '--weight_path', weight_dir,
                '--h', str(max_h),
                '--w', str(max_w)], stdout=subprocess.PIPE, cwd=str(Path(base_script_path, 'neuralmaterial')))

        MAT_OT_NEURAL_GetInterpolations._popen = process

        gan.progress = 'Generating interpolations ...'
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}

class MAT_OT_NEURAL_StopGenerator(Operator):
    bl_idname = "neural.stop_generator"
    bl_label = "Stop generator material"
    bl_description = "Stop generate base material from flash images."

    @classmethod
    def poll(self, context):
        return MAT_OT_NEURAL_Generator._popen
    
    def execute(self, context):
        MAT_OT_NEURAL_Generator._popen.terminate()
        return {'FINISHED'}

class MAT_OT_NEURAL_RevertMaterial(Operator):
    bl_idname = "neural.revert_material"
    bl_label = "Revert edited material"
    bl_description = "Trys to revert a material to older iteration if possible"
    
    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.neural_properties.progress and \
        os.path.isfile(os.path.join(bpy.context.scene.neural_properties.directory, 'out', 'old_render.png'))

    def preprocess(self, context):
        if bpy.context.active_object:
            name = f"{bpy.context.active_object.name}_neural"
        else:
            name = "neural"
        
        # First unlink files
        check_remove_img(f'{name}_render.png')
        check_remove_img(f'{name}_albedo.png')
        check_remove_img(f'{name}_rough.png')
        check_remove_img(f'{name}_specular.png')
        check_remove_img(f'{name}_normal.png')
    
    def execute(self, context):
        gan = bpy.context.scene.neural_properties

        self.preprocess(context)

        # Rename old files
        out = os.path.join(gan.directory, 'out')

        old_weights_path = os.path.join(out, 'old_weights.ckpt')
        old_render_path = os.path.join(out, 'old_render.png')
        old_albedo_path = os.path.join(out, 'old_albedo.png')
        old_rough_path = os.path.join(out, 'old_rough.png')
        old_specular_path = os.path.join(out, 'old_specular.png')
        old_normal_path = os.path.join(out, 'old_normal.png')


        weights_path = os.path.join(out, 'weights.ckpt')
        render_path = os.path.join(out, 'render.png')
        albedo_path = os.path.join(out, 'albedo.png')
        rough_path = os.path.join(out, 'rough.png')
        specular_path = os.path.join(out, 'specular.png')
        normal_path = os.path.join(out, 'normal.png')

        # Copy and replace old files
        shutil.move(old_weights_path, weights_path)
        shutil.move(old_render_path, render_path)
        shutil.move(old_albedo_path, albedo_path)
        shutil.move(old_rough_path, rough_path)
        shutil.move(old_specular_path, specular_path)
        shutil.move(old_normal_path, normal_path)

        # Update material textures
        update_neural(bpy.context.active_object, out)

        gan.progress = 'Material reverted'
        
        return {'FINISHED'}