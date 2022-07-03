from custom_trajectron_for_mpc import *
import sys
import time
sys.path.append('../../')

### TO RUN ###
#### open command prompt and key in conda activate trajectron++

##############################
### Generating Environment ###
##############################
env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)

#set the attention radius, which activates edges based on proximity
attention_radius = dict()
attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0 
env.attention_radius = attention_radius

## initialise dummy scene from node_df 
time_per_timestep = 0.4
scene = initialise_dummy_scene(100,0.4)
env.scenes = [scene]

## initialise and calibrate trajectron to env
new_trajectron = initialise_trajectron_calibrated_to_env(env)

## set the horizon you wish to predict until, and number of trajectories
horz = 3
num_trajs = 3

## put your x,y data in these lists below, contiuously append/ modify the lists with new info
x_pos_list2 = []
y_pos_list2 = []

## At each timestep, clear the scene & append new nodes from new lists
scene.nodes.clear()
scene.nodes.append(generate_node_from_pos_lsts(x_pos_list2,y_pos_list2,0.4,env.NodeType.PEDESTRIAN,"test"))

## predict with the new scene, producing lists of all mean x,y ; std x,y 
mean_x, std_x, mean_y, std_y = predict_from_timestep(new_trajectron,horz,num_trajs,2,scene)
print(mean_x,std_x,mean_y,std_y)