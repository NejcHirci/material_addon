# The following script contains classes with necessary Blender operators
# for the algorhtmic mix approach in material generation and editing.
#
# Node setup for generating PBR maps from albedo images was developed by cgvirus
# and is available on the following repository: https://github.com/cgvirus/photo-to-pbr-texture-blender

import os
from pathlib import Path
from threading import Thread
import time

import bpy
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

def check_remove_img(name):
    if name in bpy.data.images:
        image = bpy.data.images[name]
        bpy.data.images.remove(image)

# Function for updating textures during material generation.
def update_mix(base_path):
    # Update textures if they already exist
    active_obj = bpy.context.view_layer.objects.active

    if active_obj:
        base_name = f"{active_obj.name}_mix_mat"
        if base_name not in bpy.data.materials:
            mat = bpy.data.materials["mix_mat"].copy()
            mat.name = base_name
        else:
            mat = bpy.data.materials[base_name]
    else:
        base_name = "base"
        mat = bpy.data.materials["mix_mat"]
        
    nodes = mat.node_tree.nodes

    albedo = nodes.get("Image Texture")
    metallic = nodes.get("Image Texture.001")
    ambient_occlusion = nodes.get("Image Texture.002")
    cavity = nodes.get("Image Texture.004")
    normal = nodes.get("Image Texture.005")
    roughness = nodes.get("Image Texture.006")
    emission = nodes.get("Image Texture.007")

    if os.path.isfile(os.path.join(base_path, 'Albedo0000.png')):
        check_remove_img(f'{base_name}-mixmat-albedo.png')
        img = bpy.data.images.load(os.path.join(base_path, 'Albedo0000.png'))
        img.name = f'{base_name}-mixmat-albedo.png'
        albedo.image = img
        check_remove_img(f'{base_name}-mixmat-metallic.png')
        img = bpy.data.images.load(os.path.join(base_path, 'Metallic0000.png'))
        img.name = f'{base_name}-mixmat-metallic.png'
        metallic.image = img    
        check_remove_img(f'{base_name}-mixmat-rough.png')
        img = bpy.data.images.load(os.path.join(base_path, 'Roughness0000.png'))
        img.name = f'{base_name}-mixmat-rough.png'
        roughness.image = img
        check_remove_img(f'{base_name}-mixmat-normal.png')
        img = bpy.data.images.load(os.path.join(base_path, 'Normal0000.png'))
        img.name = f'{base_name}-mixmat-normal.png'
        normal.image = img
        check_remove_img(f'{base_name}-mixmat-ao.png')
        img = bpy.data.images.load(os.path.join(base_path, 'AO0000.png'))
        img.name = f'{base_name}-mixmat-ao.png'
        ambient_occlusion.image = img
        check_remove_img(f'{base_name}-mixmat-cavity.png')
        img = bpy.data.images.load(os.path.join(base_path, 'Cavity0000.png'))
        img.name = f'{base_name}-mixmat-cavity.png'
        cavity.image = img
        check_remove_img(f'{base_name}-mixmat-emission.png')
        img = bpy.data.images.load(os.path.join(base_path, 'Emission0000.png'))
        img.name = f'{base_name}-mixmat-emission.png'
        emission.image = img

class MAT_OT_MIX_Generator(Operator):
    bl_idname = "mixmat.generator"
    bl_label = "Generate PBR maps from albedo"
    bl_description = "Generate PBR maps from albedo"

    @classmethod
    def poll(self, context):
        return "Ready" in bpy.context.scene.mixmat_properties.progress

    def execute(self, context):
        mixmat = bpy.context.scene.mixmat_properties

        sTime = time.time()

        image_dirs = [str(p) for p in Path(mixmat.directory).iterdir() if p.is_file() and p.suffix == '.png']

        out_dir = Path(mixmat.directory, 'out')
        out_dir.mkdir(parents=True, exist_ok=True)

        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree

        for node in tree.nodes:
            tree.nodes.remove(node)

        photo_to_pbr = tree.nodes.new(type='CompositorNodeGroup')
        photo_to_pbr.node_tree = bpy.data.node_groups['photo_to_pbr']

        main_input = photo_to_pbr.node_tree.nodes['Image']
        check_remove_img('mixmat-input.png')
        img = bpy.data.images.load(image_dirs[0])
        img.name = 'mixmat-input.png'
        main_input.image = img

        main_output = photo_to_pbr.node_tree.nodes['File Output.001']
        main_output.base_path = str(out_dir)

        # Because of naming we must reset current frame.
        bpy.context.scene.frame_set(0)

        mixmat.progress = "Generating textures from albedo ..."

        for area in context.screen.areas:
            if area.type in ['NODE_EDITOR']:
                area.tag_redraw()

        bpy.context.window.cursor_set("WAIT")

        bpy.ops.render.render()

        for node in bpy.context.scene.node_tree.nodes:
            bpy.context.scene.node_tree.nodes.remove(node)

        out_dir = Path(mixmat.directory, 'out')
        bpy.context.scene.use_nodes = False

        update_mix(str(out_dir))
        mixmat.progress = "Textures generated."
        mixmat.progress += f" Elapsed time: {time.time()-sTime:.3f}"

        bpy.context.window.cursor_set("DEFAULT")

        return {'FINISHED'}

class MAT_OT_MIX_FileBrowser(Operator, ImportHelper):
    """File browser operator"""
    bl_idname= "mixmat.file_browser"
    bl_label = "Selects folder with data"
    
    filename_ext = ""

    def execute(self, context):
        mixmat = bpy.context.scene.mixmat_properties
        fdir = self.properties.filepath
        mixmat.directory = os.path.dirname(fdir)
        fdir = os.path.dirname(fdir)
        
        if os.path.isdir(os.path.join(fdir, 'out')):
            bpy.context.scene.matgan_properties.progress = "Material found."
            update_mix(os.path.join(fdir, 'out'))        
        else:
            bpy.context.scene.matgan_properties.progress = "Ready to generate."
        return {'FINISHED'}