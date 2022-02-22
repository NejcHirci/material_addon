import bpy

def update_mixmat_interpolate(self, context):
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
    mix_shader = nodes.get("Mix Shader")
    mix_shader.inputs[0].default_value = 1.0 - context.scene.mixmat_properties.value

def update_mixmat_direction(self, context):
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
    links = mat.node_tree.links
    mix_shader = nodes.get("Mix Shader")
    group = nodes.get("Group.001")
    group.node_tree = bpy.data.node_groups.get(context.scene.mixmat_properties.material)
    links.new(group.outputs[0], mix_shader.inputs[1])

    bpy.data.materials[context.scene.mixmat_properties.material].preview.reload()


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
        default=1024, min=512, max=3000)
    w_res : bpy.props.IntProperty(name="Super resolution width", subtype="PIXEL", description="Width resolution for upscaling", \
        default=1024, min=512, max=3000)
    seed : bpy.props.IntProperty(name="Seed", description="Seed used for material generation", default=42, min=0, max=100000)

class MixMaterialProps(bpy.types.PropertyGroup):
    directory: bpy.props.StringProperty(name="Import folder", description="The folder to import images from",
        default="")
    progress: bpy.props.StringProperty(name="Progress value", description="", default="Not started.",
        options={'SKIP_SAVE'})
    material: bpy.props.EnumProperty(name="Preset materials", description="", items= { 
            ('Aluminium', 'Brushed Aluminium', 'Downloaded from https://www.blenderkit.com/get-blenderkit/738656a5-5912-42a3-bf0e-91b604b4625c/'),
            ('Wood', 'Wood', 'Downloaded from https://www.blenderkit.com/get-blenderkit/cf759562-e8e9-4f86-98eb-13c1975389ad/'),
            ('Plastic', 'Plastic touched', 'Downlaoded from https://www.blenderkit.com/get-blenderkit/d52c1cd6-b74c-4b80-8b94-4e68ed471053/'),
            ('Plaster', 'Plaster', 'Downloaded from https://www.blenderkit.com/get-blenderkit/6f5946c7-5ad5-4fee-8ff4-7783108f3939/'),
            ('Leather', 'Leather', 'Downloaded from https://www.blenderkit.com/get-blenderkit/4ca36a05-02b4-4c76-a0ae-1b65cf6fd678/'),
            ('Silk', 'Silk', 'Downloaded from https://www.blenderkit.com/get-blenderkit/85c6a076-4967-4563-8662-f677777f6d3a/'),
            ('Concrete', 'Concrete', 'Downloaded from https://www.blenderkit.com/get-blenderkit/0662b3bf-a762-435d-9407-e723afd5eafc/'),
            ('Marble', 'Marble', 'Downloaded from https://www.blenderkit.com/get-blenderkit/9cb2b199-f3fe-4cbc-b963-cf7d1332a87d/'),
        },
        default="Wood", update=update_mixmat_direction)
    value: bpy.props.FloatProperty(name="Mix shader value", description="", default=0.0, min=0.0, max=1.0, update=update_mixmat_interpolate, options={'SKIP_SAVE'})

def register():
    bpy.types.Scene.matgan_properties = bpy.props.PointerProperty(type=MaterialGANProps)
    bpy.types.Scene.neural_properties = bpy.props.PointerProperty(type=NeuralMaterialProps)
    bpy.types.Scene.mixmat_properties = bpy.props.PointerProperty(type=MixMaterialProps)

def unregister():
    del bpy.types.Scene.matgan_properties
    del bpy.types.Scene.neural_properties
    del bpy.types.Scene.mixmat_properties
