import mujoco as mj
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
from mujoco.glfw import glfw

# Load the MuJoCo model from an XML file
is_windows = sys.platform.startswith('win')

if is_windows:
    xml_path = 'C:/Users/NidhiParayil/nidhi/mujoco_reactive_controller/assets/franka_emika_panda/panda.xml'
else:
    xml_path = "/home/nidhi/MPC_python/mujoco_reactive_controller/assets/franka_emika_panda/panda.xml"



model = mj.MjModel.from_xml_path(xml_path) 
data = mj.MjData(model)    



# install GLFW mouse and keyboard callbacks
# glfw.set_key_callback(window, keyboard)
# glfw.set_cursor_pos_callback(window, mouse_move)
# glfw.set_mouse_button_callback(window, mouse_button)
# glfw.set_scroll_callback(window, scroll)

# Joint names to control with sliders
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]  # Replace with actual joint names in your model
joint_ids =  [model.jnt(joint).id for joint in joint_names]
joint_indices = [data.qpos[j] for j in joint_ids]

# Number of joints
n_joints = len(joint_names)

duration = 10  # (seconds)
framerate = 60  # (Hz)
frames = []

# enable joint visualization option:
scene_option = mj.MjvOption()
scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
# Function to update the simulation based on slider values
# def update(val):
#     for i, joint_index in enumerate(joint_indices):
#         data.qpos[joint_index] = sliders[i].val
#     mj.mj_step(model,data)
#     viewer.render()

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)
with mujoco.Renderer(model) as renderer:
  while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data, scene_option=scene_option)
      pixels = renderer.render()
      frames.append(pixels)

# media.show_video(frames, fps=framerate)