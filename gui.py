import bpy
import glob
from bpy.types import Panel, Operator
from bpy.app.handlers import persistent
import os
import threading
from queue import Queue
from pathlib import Path

from . mix_ops import *
from . matgan_ops import *
from . neural_ops import *

cache_path = os.path.join(Path(__file__).parent.resolve(), '.cache')

# Redraw all function
def redraw_all(context):
    for area in context.screen.areas:
        if area.type in ['NODE_EDITOR']:
            area.tag_redraw()

# Thread function for reading output
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line.decode('utf-8').strip())
    out.close()

@persistent
def on_addon_load(dummy):
    MAT_OT_MATGAN_GetInterpolations._popen = None
    MAT_OT_MATGAN_Generator._popen = None
    MAT_OT_MATGAN_InputFromFlashImage._popen = None
    MAT_OT_MATGAN_SuperResolution._popen = None

    bpy.context.scene.matgan_properties.directory = bpy.path.abspath("//") + 'matgan_demos\\'
    bpy.context.scene.neural_properties.directory = bpy.path.abspath("//") + 'neural_demos\\'
    bpy.context.scene.mixmat_properties.directory = bpy.path.abspath("//") + 'mixmat_demos\\'

    blender_path = os.path.join(Path(__file__).parent.resolve(), 'final.blend')
    with bpy.data.libraries.load(blender_path, link=False) as (data_from, data_to):
        data_to.materials = [mat for mat in data_from.materials if mat not in data_to.materials]

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    else:
        for root, dirs, files in os.walk(cache_path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

def fix_missing(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'TEX_IMAGE':
            img = n.image
            if not img.has_data:
                img = bpy.data.images.load(os.path.join(Path(__file__).parent.resolve(), 'blank.jpg'))
                n.image = img

def update_active_mat(self, context):
    active_obj = bpy.context.view_layer.objects.active
    if active_obj:
        if context.scene.SelectWorkflow == 'MatGAN':
            base_name = "matgan_mat"
        elif context.scene.SelectWorkflow == 'NeuralMAT':
            base_name = "neural_mat"
        elif context.scene.SelectWorkflow == 'MixMAT':
            base_name = "mix_mat"

        name = f"{active_obj.name}_{base_name}"

        if name not in bpy.data.materials:
            mat = bpy.data.materials[base_name].copy()
            mat.name = name
        else:
            mat = bpy.data.materials[name]
        active_obj.active_material = mat
        fix_missing(mat)

# Copy files to .cache folder
def copy_to_cache(src_path):
    if os.path.isdir(src_path):
        for file in os.listdir(os.fsencode(src_path)):
            f = os.fsdecode(file)
            if f.endswith(".png") or f.endswith(".pt") or f.endswith('.ckpt'): 
                shutil.copyfile(os.path.join(src_path, f), os.path.join(cache_path, f))

def register():
    if on_addon_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_addon_load)

    bpy.types.Scene.SelectWorkflow = bpy.props.EnumProperty(
        name='Material System Select',
        description='Selected Material System for editing and generation.',
        items={
            ('MatGAN', 'MaterialGAN + LIIF', 'Using MaterialGAN for generation and LIIF model for upscaling. ' \
                + 'Editing implemented as vector space exploration.'), 
            ('NeuralMAT', 'Neural Material', 'Using Neural Material model for generatiog. ' \
                + 'Editing implemented as material interpolations.'), 
            ('MixMAT', 'Algorithmic generation', 'Using a Blender shader nodes approach for ' \
                + 'generating textures from albedo with mix blender shader nodes for editing.')
        },
        default='MatGAN',
        update=update_active_mat,
        options={'SKIP_SAVE'}
    )

def unregister():
    if on_addon_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_addon_load)


class MAT_PT_GeneratorPanel(Panel):
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_label = "Modifier operations"
    bl_category = "MaterialGenerator Util"

    thumb_scale = 8.0
    check_existing = False
    mix_preview = None

    def draw_matgan(self, context):
        layout = self.layout
        matgan = bpy.context.scene.matgan_properties

        # ================================================
        # Draw MaterialGAN props and operators
        # ================================================
        
        row = layout.row()
        row.prop(matgan, "progress", emboss=False, text="Status")
        
        row = layout.row()
        col = row.column()
        col.prop(matgan, "num_rend", text="Num of images")
        col = row.column()
        col.prop(matgan, "epochs", text="Epochs")
        
        row = layout.row()
        row.prop(matgan, "directory", text="Directory")
        row.operator("matgan.file_browser", icon="FILE_FOLDER", text="")

        row = layout.row()
        col = row.column()
        col.operator("matgan.input_from_images", text="Format flash images") 
        
        row = layout.row()
        col = row.column()
        col.operator("matgan.mat_from_images", text="Generate Material") 
        col = row.column()
        col.operator("matgan.stop_generator", text="", icon="PAUSE")

        layout.separator()

        # ================================================
        # Draw Upscale LIIF
        # ================================================

        row = layout.row()
        col = row.column()
        col.prop(matgan, "h_res", text="Height resolution")
        col = row.column()
        col.prop(matgan, "w_res", text="Width resolution")
        row = layout.row()
        row.operator("matgan.super_res", text="Upscale material")

        layout.separator()

        row = layout.row()
        row.operator("matgan.get_interpolations", text="Get interpolations")

        layout.separator()

        
        # ================================================
        # Draw Gallery view
        # ================================================

        if MAT_OT_MATGAN_GetInterpolations._popen is None and MAT_OT_MATGAN_Generator._popen is None:
            self.draw_gallery(context, matgan, "matgan")
        
    def draw_gallery(self, context, gan, mode):
        x = MAT_OT_GalleryDirection.direction
        interp_dir = os.path.join(gan.directory, 'interps')
        out_dir =  os.path.join(gan.directory, 'out')

        if f'7_{x}_render.png' in bpy.data.images and f'{mode}-render.png' in bpy.data.images:
            layout = self.layout
            row = layout.row()
            sign = '+' if MAT_OT_GalleryDirection.direction == 1 else '-'
            row.operator("wm.edit_direction_toggle", text="Toggle direction")

            box = layout.box()
            cols = box.column_flow(columns=3)

            # Get images
            dir_list = sorted(glob.glob(interp_dir + f'/*_{x}_render.png'))

            id = 0
            for dir in dir_list:
                if id == 4:
                    in_box = cols.box()
                    col = in_box.column()
                    img = bpy.data.images[f'{mode}-render.png']
                    img.preview_ensure()
                    col.template_icon(icon_value=img.preview.icon_id, scale=10)
                    col.label(text="Current material")
                name = os.path.split(dir)[1]
                img = bpy.data.images[name]
                img.preview_ensure()
                in_box = cols.box()
                col = in_box.column()
                col.template_icon(icon_value=img.preview.icon_id, scale=10)
                operator = col.operator(f'{mode}.edit_move', text=f"Semantic {sign}{name[0]}")
                operator.direction = name[0]
                id += 1
            
    def draw_neural(self, context):
        layout = self.layout
        neural = bpy.context.scene.neural_properties

        # ================================================
        # Draw NeuralMaterial props and operators
        # ================================================

        row = layout.row()
        row.prop(neural, "progress", emboss=False, text="Status")
        
        row = layout.row()
        col = row.column()
        col.prop(neural, "num_rend", text="Images")
        col = row.column()
        col.prop(neural, "epochs", text="Epochs")
        col = row.column()
        col.prop(neural, "seed", text="Seed")
        
        row = layout.row()
        col = row.column()
        col.prop(neural, "h_res", text="Height resolution")
        col = row.column()
        col.prop(neural, "w_res", text="Width resolution")
        
        row = layout.row()
        row.prop(neural, "directory", text="Directory")
        row.operator("neural.file_browser", icon="FILE_FOLDER", text="")

        row = layout.row()
        col = row.column()
        col.operator("neural.generator", text="Generate Material") 
        col = row.column()
        col.operator("neural.stop_generator", text="", icon="PAUSE")
        row = layout.row()
        col = row.column()
        col.operator("neural.reseed", text="Reseed Material")

        layout.separator()

        # ================================================
        # Draw NeuralMaterial interpolations operator
        # ================================================

        row = layout.row()
        row.operator("neural.get_interpolations", text="Get interpolations")

        layout.separator()

        # ================================================
        # Draw Gallery view
        # ================================================

        if MAT_OT_NEURAL_GetInterpolations._popen is None and MAT_OT_NEURAL_Generator._popen is None:
            self.draw_gallery(context, neural, "neural")

    def draw_mixmat(self, context):
        layout = self.layout
        mix = bpy.context.scene.mixmat_properties

        # ================================================
        # Draw Mix Materials generator operator
        # ================================================

        row = layout.row()
        row.prop(mix, "progress", emboss=False, text="Status")

        row = layout.row()
        row.prop(mix, "directory", text="Directory")
        row.operator("mixmat.file_browser", icon="FILE_FOLDER", text="")

        row = layout.row()
        row.operator("mixmat.generator", text="Generate")

        layout.separator()

        # ================================================
        # Draw Mix material interpolations operator
        # ================================================

        row = layout.row()
        row.prop(mix, "material", text="Select")
        row.prop(mix, "value", text="Mix level")
        

        if 'generated' in mix.progress:
            layout.separator()
            row = layout.row()
            MAT_PT_GeneratorPanel.mix_preview = row.template_preview(bpy.data.materials[mix.material], show_buttons=False)

    def draw(self, context):        
        self.layout.prop(context.scene, 'SelectWorkflow')
        if context.scene.SelectWorkflow == 'MatGAN':
            self.draw_matgan(context)
        elif context.scene.SelectWorkflow == 'NeuralMAT':
            self.draw_neural(context)
        elif context.scene.SelectWorkflow == 'MixMAT':
            self.draw_mixmat(context)
        
class MAT_OT_StatusUpdater(Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_status_updater"
    bl_label = "Modal Status Updater"

    _sTime = 0
    _timer = None
    _thread = None
    _q = Queue()

    def modal(self, context, event):
        gan = bpy.context.scene.matgan_properties
        
        if event.type == 'TIMER':                
            if MAT_OT_MATGAN_Generator._popen:
                if MAT_OT_MATGAN_Generator._popen.poll() is None:
                    try:
                        line = self._q.get_nowait()
                        print(line)
                        update_matgan(os.path.join(gan.directory, 'out'))
                        gan.progress = line
                        gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                        redraw_all(context)
                    except:
                        pass
                else:
                    copy_to_cache(os.path.join(gan.directory, 'out'))
                    update_matgan(cache_path)
                    gan.progress = "Material generated."
                    redraw_all(context)
                    MAT_OT_MATGAN_Generator._popen = None
                    self.cancel(context)
                    gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                    return {'CANCELLED'}

            elif MAT_OT_MATGAN_InputFromFlashImage._popen:
                if MAT_OT_MATGAN_InputFromFlashImage._popen.poll() is None:
                    try:
                        line = self._q.get_nowait()
                        print(line)
                        gan.progress = line
                        gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                        redraw_all(context)
                    except:
                        pass
                else:
                    gan.progress = "Input ready."
                    gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                    redraw_all(context)
                    MAT_OT_MATGAN_InputFromFlashImage._popen = None
                    self.cancel(context)
                    return {'CANCELLED'}
            
            elif MAT_OT_MATGAN_SuperResolution._popen:
                if MAT_OT_MATGAN_SuperResolution._popen.poll() is not None:
                    gan.progress = "Material upscaled."
                    copy_to_cache(os.path.join(gan.directory, 'out'))
                    update_matgan(cache_path)
                    redraw_all(context)
                    MAT_OT_MATGAN_SuperResolution._popen = None
                    self._thread = None
                    self.cancel(context)
                    gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                    return {'CANCELLED'}
            
            elif MAT_OT_MATGAN_GetInterpolations._popen:
                if MAT_OT_MATGAN_GetInterpolations._popen.poll() is None:
                    try:
                        line = self._q.get_nowait()
                        print(line)
                        gan.progress = line
                        gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                        redraw_all(context)
                    except:
                        pass
                else:
                    check_remove_img('matgan-render.png')
                    img = bpy.data.images.load(os.path.join(gan.directory, 'out') + '/render.png')
                    img.name = 'matgan-render.png'

                    interp_path = os.path.join(gan.directory, 'interps')
                    dir_list = sorted(glob.glob(interp_path + '/*_*_render.png'))
                    for dir in dir_list:
                        check_remove_img(os.path.split(dir)[1])
                        img = bpy.data.images.load(dir)
                        img.name = os.path.split(dir)[1]
                    gan.progress = "Material interpolations generated."
                    gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                    copy_to_cache(os.path.join(gan.directory, 'out'))
                    update_matgan(cache_path)
                    redraw_all(context)
                    MAT_OT_MATGAN_GetInterpolations._popen = None
                    self.cancel(context)
                    return {'CANCELLED'}

            elif MAT_OT_NEURAL_Generator._popen:
                gan = bpy.context.scene.neural_properties
                if MAT_OT_NEURAL_Generator._popen.poll() is None:
                    try:
                        line = self._q.get_nowait()
                        print(line)
                        update_neural(os.path.join(gan.directory, 'out'))
                        gan.progress = line
                        gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                        redraw_all(context)
                    except:
                        pass
                else:
                    copy_to_cache(os.path.join(gan.directory, 'out'))
                    update_neural(cache_path)
                    gan.progress = "Material generated."
                    gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                    redraw_all(context)
                    MAT_OT_NEURAL_Generator._popen = None
                    self.cancel(context)
                    return {'CANCELLED'}

            elif MAT_OT_NEURAL_GetInterpolations._popen:
                gan = bpy.context.scene.neural_properties
                if MAT_OT_NEURAL_GetInterpolations._popen.poll() is None:
                    try:
                        line = self._q.get_nowait()
                        print(line)
                        gan.progress = line
                        gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                        redraw_all(context)
                    except:
                        pass
                else:
                    check_remove_img('neural-render.png')
                    img = bpy.data.images.load(os.path.join(gan.directory, 'out') + '/render.png')
                    img.name = 'neural-render.png'

                    interp_path = os.path.join(gan.directory, 'interps')
                    dir_list = sorted(glob.glob(interp_path + '/*_*_render.png'))
                    for dir in dir_list:
                        check_remove_img(os.path.split(dir)[1])
                        img = bpy.data.images.load(dir)
                        img.name = os.path.split(dir)[1]
                    gan.progress = "Material interpolations generated."
                    gan.progress += f" Elapsed time: {time.time()-self._sTime:.3f}"
                    copy_to_cache(os.path.join(gan.directory, 'out'))
                    update_neural(cache_path)
                    redraw_all(context)
                    MAT_OT_NEURAL_GetInterpolations._popen = None
                    self.cancel(context)
                    return {'CANCELLED'}

            else:
                self.cancel(context)
                return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def execute(self, context):
        self._sTime = time.time()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        if MAT_OT_MATGAN_Generator._popen:
            self._thread = threading.Thread(target=enqueue_output, args=(MAT_OT_MATGAN_Generator._popen.stdout, self._q), daemon=True)
        elif MAT_OT_MATGAN_InputFromFlashImage._popen:
            self._thread = threading.Thread(target=enqueue_output, args=(MAT_OT_MATGAN_InputFromFlashImage._popen.stdout, self._q), daemon=True)
        elif MAT_OT_MATGAN_GetInterpolations._popen:
            self._thread = threading.Thread(target=enqueue_output, args=(MAT_OT_MATGAN_GetInterpolations._popen.stdout, self._q), daemon=True)
        elif MAT_OT_MATGAN_SuperResolution._popen:
            self._thread = threading.Thread(target=enqueue_output, args=(MAT_OT_MATGAN_SuperResolution._popen.stdout, self._q), daemon=True)
        elif MAT_OT_NEURAL_Generator._popen:
            self._thread = threading.Thread(target=enqueue_output, args=(MAT_OT_NEURAL_Generator._popen.stdout, self._q), daemon=True)
        elif MAT_OT_NEURAL_GetInterpolations._popen:
            self._thread = threading.Thread(target=enqueue_output, args=(MAT_OT_NEURAL_GetInterpolations._popen.stdout, self._q), daemon=True)
        self._thread.start()
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

class MAT_OT_GalleryDirection(Operator):
    """Operator which switches gallery edit direction"""
    bl_idname = "wm.edit_direction_toggle"
    bl_label = "Direction switch operator"


    direction = 1

    def execute(self, context):
        if MAT_OT_GalleryDirection.direction == 1:
            MAT_OT_GalleryDirection.direction = 2
        else:
            MAT_OT_GalleryDirection.direction = 1
        return {'FINISHED'}
