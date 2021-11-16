# The following script contains classes with necessary Blender operators
# for the algorhtmic mix approach in material generation and editing.
#
# Node setup for generating PBR maps from albedo images was developed by cgvirus
# and is available on the following repository: https://github.com/cgvirus/photo-to-pbr-texture-blender

import os
from pathlib import Path

import bpy
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

def check_remove_img(name):
    if name in bpy.data.images:
        image = bpy.data.images[name]
        bpy.data.images.remove(image)

# Function for updating textures during material generation.
def update_neural(base_path):
    # Update textures if they already exist
    mat = bpy.data.materials["mixmat_mat"]
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

class MAT_OT_MIX_Generator(Operator):
    bl_idname = "mixmat.generator"
    bl_label = "Generate PBR maps from albedo"
    bl_description = "Generate PBR maps from albedo"

    @classmethod
    def poll(self, context):
        return "Ready" in bpy.context.scene.mixmat_properties.progress

    def execute(self, context):
        mixmat = bpy.context.scene.mixmat_properties

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


        bpy.ops.render.render()
 
        for node in tree.nodes:
            tree.nodes.remove(node)
            
        bpy.context.scene.use_nodes = False

        mixmat.progress = "Textures generated"
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
        mixmat.progress = "Ready to generate."
        return {'FINISHED'}