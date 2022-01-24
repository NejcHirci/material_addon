# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from struct import pack
import bpy
import sys
import os

from . import addon_updater_ops
from . import gui
from . import matgan_ops
from . import mix_ops
from . import neural_ops
from . import props

bl_info = {
    "name": "material_addon",
    "author": "Nejc Hirci",
    "description": "Provides 3 approaches to material generation from photos.",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "ShaderEditor > Util > MaterialGenerator Util",
    "doc_url": "https://github.com/NejcHirci/material-addon",
    "warning": "",
    "category": "Material"
}

packages = [
    ("torch", "-f https://download.pytorch.org/whl/cu113/torch_stable.html torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113"),
    ("cv2", "opencv-contrib-python"),
    ("matplotlib", "matplotlib"),
    ("skimage", "scikit-image"),
    ("IPython", "ipython"),
    ("tqdm", "tqdm"),
    ("kornia", "--only-binary=numpy kornia"),
    ("yaml", "PyYaml"),
    ("hydra", "hydra-core"),
    ("tensorboard", "tensorboard")
]

class DemoUpdaterPanel(bpy.types.Panel):
    """Panel to demo popup notice and ignoring functionality"""
    bl_label = "Updater Demo Panel"
    bl_idname = "OBJECT_PT_DemoUpdaterPanel_hello"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS' if bpy.app.version < (2, 80) else 'UI'
    bl_context = "objectmode"
    bl_category = "Tools"

    def draw(self, context):
        layout = self.layout

        # Call to check for update in background.
        # Note: built-in checks ensure it runs at most once, and will run in
        # the background thread, not blocking or hanging blender.
        # Internally also checks to see if auto-check enabled and if the time
        # interval has passed.
        addon_updater_ops.check_for_update_background()

        layout.label(text="Demo Updater Addon")
        layout.label(text="")

        col = layout.column()
        col.scale_y = 0.7
        col.label(text="If an update is ready,")
        col.label(text="popup triggered by opening")
        col.label(text="this panel, plus a box ui")

        # Could also use your own custom drawing based on shared variables.
        if addon_updater_ops.updater.update_ready:
            layout.label(text="Custom update message", icon="INFO")
        layout.label(text="")

        # Call built-in function with draw code/checks.
        addon_updater_ops.update_notice_box_ui(self, context)


@addon_updater_ops.make_annotations
class DemoPreferences(bpy.types.AddonPreferences):
    """Demo bare-bones preferences"""
    bl_idname = __package__

    # Addon updater preferences.

    auto_check_update = bpy.props.BoolProperty(
        name="Auto-check for Update",
        description="If enabled, auto-check for updates using an interval",
        default=False)

    updater_interval_months = bpy.props.IntProperty(
        name='Months',
        description="Number of months between checking for updates",
        default=0,
        min=0)

    updater_interval_days = bpy.props.IntProperty(
        name='Days',
        description="Number of days between checking for updates",
        default=7,
        min=0,
        max=31)

    updater_interval_hours = bpy.props.IntProperty(
        name='Hours',
        description="Number of hours between checking for updates",
        default=0,
        min=0,
        max=23)

    updater_interval_minutes = bpy.props.IntProperty(
        name='Minutes',
        description="Number of minutes between checking for updates",
        default=0,
        min=0,
        max=59)

    def draw(self, context):
        layout = self.layout

        # Works best if a column, or even just self.layout.
        mainrow = layout.row()
        col = mainrow.column()

        # Updater draw function, could also pass in col as third arg.
        addon_updater_ops.update_settings_ui(self, context)

        # Alternate draw function, which is more condensed and can be
        # placed within an existing draw function. Only contains:
        #   1) check for update/update now buttons
        #   2) toggle for auto-check (interval will be equal to what is set above)
        # addon_updater_ops.update_settings_ui_condensed(self, context, col)

        # Adding another column to help show the above condensed ui as one column
        # col = mainrow.column()
        # col.scale_y = 2
        # ops = col.operator("wm.url_open","Open webpage ")
        # ops.url=addon_updater_ops.updater.website

classes = (
    DemoPreferences,
    DemoUpdaterPanel,
    props.MixMaterialProps,
    mix_ops.MAT_OT_MIX_Generator,
    neural_ops.MAT_OT_NEURAL_Generator,
    matgan_ops.MAT_OT_MATGAN_InputFromFlashImage,
    neural_ops.MAT_OT_NEURAL_GetInterpolations,
    matgan_ops.MAT_OT_MATGAN_GetInterpolations,
    props.NeuralMaterialProps,
    gui.MAT_OT_StatusUpdater,
    props.MaterialGANProps,
    mix_ops.MAT_OT_MIX_FileBrowser,
    matgan_ops.MAT_OT_MATGAN_Generator,
    matgan_ops.MAT_OT_MATGAN_StopGenerator,
    matgan_ops.MAT_OT_MATGAN_SuperResolution,
    neural_ops.MAT_OT_NEURAL_FileBrowser,
    neural_ops.MAT_OT_NEURAL_Reseed,
    gui.MAT_OT_GalleryDirection,
    matgan_ops.MAT_OT_MATGAN_FileBrowser,
    neural_ops.MAT_OT_NEURAL_StopGenerator,
    matgan_ops.MAT_OT_MATGAN_EditMove,
    neural_ops.MAT_OT_NEURAL_EditMove,
    gui.MAT_PT_GeneratorPanel,
)

def register():
    addon_updater_ops.register(bl_info)

    for cls in classes:
        bpy.utils.register_class(cls)

    gui.register()
    props.register()

def unregister():
    addon_updater_ops.unregister()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    gui.unregister()
    props.unregister()
