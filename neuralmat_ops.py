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

def check_remove_img(name):
    if name in bpy.data.images:
        image = bpy.data.images[name]
        bpy.data.images.remove(image)

# Function for updating textures during material generation.
def update_neural(base_path):
    # Update textures if they already exist
    mat = bpy.data.materials["neural_mat"]
    nodes = mat.node_tree.nodes

    albedo = nodes.get("Image Texture")
    specular = nodes.get("Image Texture.001")
    rough = nodes.get("Image Texture.002")
    normal = nodes.get("Image Texture.003")

    if os.path.isfile(os.path.join(base_path, 'albedo.png')):
        check_remove_img('neural-render.png')
        img = bpy.data.images.load(os.path.join(base_path, 'render.png'))
        img.name = 'neural-render.png'
        check_remove_img('neural-albedo.png')
        img = bpy.data.images.load(os.path.join(base_path, 'albedo.png'))
        img.name = 'neural-albedo.png'
        albedo.image = img
        check_remove_img('neural-specular.png')
        img = bpy.data.images.load(os.path.join(base_path, 'specular.png'))
        img.name = 'neural-specular.png'
        specular.image = img    
        check_remove_img('neural-rough.png')
        img = bpy.data.images.load(os.path.join(base_path, 'rough.png'))
        img.name = 'neural-rough.png'
        rough.image = img
        check_remove_img('neural-normal.png')
        img = bpy.data.images.load(os.path.join(base_path, 'normal.png'))
        img.name = 'neural-normal.png'
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
    bl_idname = "neuralmat.get_interpolations"
    bl_label = "Get interpolations"
    bl_description = "Generate interpolations in discovered directions"

    _popen = None

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.neuralmat_properties.progress

    def execute(self, context):
        neuralmat = bpy.context.scene.neuralmat_properties

        in_dir  = neuralmat.directory
        weight_dir = os.path.join(neuralmat.directory, 'out', 'weights.ckpt')

        model_path = './trainings/Neuralmaterial'

        # Call to generate texture maps
        python_exe = sys.executable
        process = subprocess.Popen([python_exe, '-u', './scripts/get_interpolations.py',
                '--model', model_path,
                '--input_path', in_dir,
                '--weight_path', weight_dir,
                '--h', str(neuralmat.h_res),
                '--w', str(neuralmat.w_res)], stdout=subprocess.PIPE, cwd=str(Path(base_script_path, 'neuralmaterial')))

        MAT_OT_NEURAL_GetInterpolations._popen = process

        neuralmat.progress = 'Generating interpolations ...'
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}

class MAT_OT_NEURAL_FileBrowser(Operator, ImportHelper):
    """File browser operator"""
    bl_idname= "neuralmat.file_browser"
    bl_label = "Selects folder with data"
    
    filename_ext = ""

    def execute(self, context):
        fdir = self.properties.filepath
        gan = bpy.context.scene.neuralmat_properties
        gan.directory = os.path.dirname(fdir)
        fdir = os.path.dirname(fdir)
        if os.path.isdir(os.path.join(fdir, 'out')):
            gan.progress = "Material found."
            update_neural(os.path.join(fdir, 'out'))        
        else:
            gan.progress = "Ready to generate."
        return {'FINISHED'}

class MAT_OT_NEURAL_Generator(Operator):
    bl_idname = "neuralmat.generator"
    bl_label = "Generate neural material"
    bl_description = "Generate base material from flash images"

    _popen = None

    @classmethod
    def poll(self, context):
        return MAT_OT_NEURAL_Generator._popen is None and MAT_OT_NEURAL_GetInterpolations._popen is None

    def execute(self, context):
        neuralmat = bpy.context.scene.neuralmat_properties

        in_dir  = neuralmat.directory
        out_dir = os.path.join(neuralmat.directory, 'out')

        model_path = './trainings/Neuralmaterial'
        N = neuralmat.num_rend
        epochs = str(neuralmat.epochs)

        # Call to generate texture maps
        python_exe = sys.executable
        process = subprocess.Popen([python_exe, '-u', './scripts/test.py',
                '--model', model_path,
                '--input_path', in_dir,
                '--output_path', out_dir,
                '--epochs', epochs,
                '--h', str(neuralmat.h_res),
                '--w', str(neuralmat.w_res)], stdout=subprocess.PIPE, cwd=str(Path(base_script_path, 'neuralmaterial')))

        MAT_OT_NEURAL_Generator._popen = process

        neuralmat.progress = 'Epoch: [{}/{}]   Loss: 0.0 0.0'.format(1, neuralmat.epochs)
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}

class MAT_OT_NEURAL_EditMove(Operator):
    bl_idname = "neural.edit_move"
    bl_label = "Move material in desired material directions."
    bl_description = "Finds 9 neighbouring materials in latent space directions from the newly chosen one."

    direction : bpy.props.StringProperty(default="")

    @classmethod
    def poll(self, context):
        return "Material" in bpy.context.scene.neuralmat_properties.progress

    def execute(self, context):
        gan = bpy.context.scene.neuralmat_properties
        interp_dir = os.path.join(gan.directory, 'interps')

        # First unlink files
        check_remove_img('neural-render.png')
        check_remove_img('neural-albedo.png')
        check_remove_img('neural-rough.png')
        check_remove_img('neural-specular.png')
        check_remove_img('neural-normal.png')

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

        in_dir  = gan.directory
        weight_dir = os.path.join(gan.directory, 'out', 'weights.ckpt')

        model_path = './trainings/Neuralmaterial'

        # Call to generate texture maps
        python_exe = sys.executable
        process = subprocess.Popen([python_exe, '-u', './scripts/get_interpolations.py',
                '--model', model_path,
                '--input_path', in_dir,
                '--weight_path', weight_dir,
                '--h', str(gan.h_res),
                '--w', str(gan.w_res)], stdout=subprocess.PIPE, cwd=str(Path(base_script_path, 'neuralmaterial')))

        MAT_OT_NEURAL_GetInterpolations._popen = process

        gan.progress = 'Generating interpolations ...'
        
        bpy.ops.wm.modal_status_updater()
        
        return {'FINISHED'}

class MAT_OT_MATGAN_StopGenerator(Operator):
    bl_idname = "neuralmat.stop_generator"
    bl_label = "Stop generator material"
    bl_description = "Stop generate base material from flash images."

    @classmethod
    def poll(self, context):
        return MAT_OT_NEURAL_Generator._popen
    
    def execute(self, context):
        MAT_OT_NEURAL_Generator._popen.terminate()
        return {'FINISHED'}