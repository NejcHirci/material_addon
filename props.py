import bpy
import os
import sys
import subprocess
    
class MaterialGANProps(bpy.types.PropertyGroup):
    num_rend: bpy.props.IntProperty( name="N", description="Number of images used for material generation",
        default=9, min=1, max=9, options={'SKIP_SAVE'})
    epochs: bpy.props.IntProperty(name="Epochs", description="Number of iterations for material generation",
        default=2000, min=100, max=10000, options={'SKIP_SAVE'})
    directory: bpy.props.StringProperty(name="Import folder", description="The folder to import images from",
        default="")
    progress: bpy.props.StringProperty(name="Progress value", description="", default="Not started.", 
        options={'SKIP_SAVE'})
    h_res : bpy.props.IntProperty(name="Super resolution height", subtype="PIXEL", description="Height resolution for upscaling", \
        default=1024, min=512, max=8096)
    w_res : bpy.props.IntProperty(name="Super resolution width", subtype="PIXEL", description="Width resolution for upscaling", \
        default=1024, min=512, max=8096)

class NeuralMaterialProps(bpy.types.PropertyGroup):
    num_rend: bpy.props.IntProperty(name="N", description="Number of images used for material generation",
        default=1, min=1, options={'SKIP_SAVE'})
    epochs: bpy.props.IntProperty(name="Epochs", description="Number of iterations for material generation",
        default=2000, min=100, max=10000, options={'SKIP_SAVE'})
    directory: bpy.props.StringProperty(name="Import folder", description="The folder to import images from",
        default="")
    progress: bpy.props.StringProperty(name="Progress value", description="", default="Not started.", 
        options={'SKIP_SAVE'})
    h_res : bpy.props.IntProperty(name="Super resolution height", subtype="PIXEL", description="Height resolution for upscaling", \
        default=1024, min=512, max=8096)
    w_res : bpy.props.IntProperty(name="Super resolution width", subtype="PIXEL", description="Width resolution for upscaling", \
        default=1024, min=512, max=8096)



def register():
    bpy.types.Scene.matgan_properties = bpy.props.PointerProperty(type=MaterialGANProps)
    bpy.types.Scene.neuralmat_properties = bpy.props.PointerProperty(type=NeuralMaterialProps)
