
import gym
import robosuite as suite


from custom_task import LiftSquareObject

from custom_gym_wrapper import CustomGymWrapper

from robosuite import load_controller_config


import numpy as np

def create_env():

  camera_quat = [0.6743090152740479, 0.21285612881183624, 0.21285581588745117, 0.6743084788322449]
  pos = [0.626,0,1.6815]
  height_vs_width_relattion = 754/449
  camera_attribs = {'fovy': 31.0350747}
  camera_h = 100
  camera_w = int(camera_h * height_vs_width_relattion)




  controller_config = load_controller_config(default_controller="JOINT_POSITION")
  env = suite.make(
      camera_pos = pos,#(1.1124,-0.046,1.615),#(1.341772827,  -0.312295471 ,  0.182150085+1.5), 
      camera_quat = camera_quat,#(0.5608417987823486, 0.4306466281414032, 0.4306466579437256, 0.5608419179916382),# frontview quat
      camera_attribs = camera_attribs,
      env_name="LiftSquareObject", # try with other tasks like "Stack" and "Door"
      robots="IIWA",  # try with other robots like "Sawyer" and "Jaco"
      gripper_types="Robotiq85Gripper",
      has_renderer=False,
      has_offscreen_renderer=True,
      use_camera_obs=True,
      camera_names =['calibrated_camera'],
      camera_widths =[camera_w],
      camera_heights=[camera_h],
      camera_depths=[True],
      use_object_obs=False,
      controller_configs=controller_config,
      control_freq = 20,
      horizon = 20,
  )
      

  gym_env = CustomGymWrapper(env,['calibrated_camera_image'])
  return gym_env




