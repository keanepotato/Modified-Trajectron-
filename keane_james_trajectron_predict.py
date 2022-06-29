#import all necessary modules
from optparse import Values
from custom_trajectron_for_mpc import *
import numpy as np
# #from numpy.random import default_rng
# from casadi import *
# from casadi.tools import *
#from CVaR_valueIteration import CVaR_space
# from GP import *
# from obstacle_traj import *
# from MDP_next_waypoint import *
import pdb
import sys
import time
sys.path.append('../../')
# import do_mpc
# from custom_trajectron_for_mpc import *
from statistics import mean,stdev

"test"


### New Code for generating dynamic obstacle ###
# starting_x = 0
# final_x = 10
# starting_y = 7
# final_y = 1
# inc = 0.35

#xs, ys, d_obs_x, d_obs_y, X = obstacle_obsrv(setup_mpc['t_step'], starting_x, final_x, starting_y, final_y, inc)
# node_df = generate_pedestrian_obstacle_for_trajectron(xs,ys)
# 42

## initialise node_df from eth txt file
# node_df = obtain_node_df_from_txt("/home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/students001_val.txt",'142')
# xs = node_df['x'].to_numpy().reshape(-1,1) 
# ys = node_df['y'].to_numpy().reshape(-1,1)

## initialise scene from node_df 
time_per_timestep = 0.4

#scene = initialise_pedestrian_scene(node_df,time_per_timestep)     ## specify the amount of time for each timestep                               
#env.scenes=[scene]


frame_id = [i for i in range(0,3)]
x_pos_list = [i for i in range(0,3)]
y_pos_list = [i for i in range(0,3)]
node_df = pd.DataFrame({'frame_id': frame_id,
                                    'type': '',
                                    #'node_id': annotation['instance_token'],
                                    'robot': False,
                                    'x': x_pos_list,
                                    'y': y_pos_list,
                                    'z': 0,
                                    'length': 0.4,
                                    'width': 0.5,
                                    'height': 0.3,
                                    'heading': 0})

scene = initialise_pedestrian_scene(node_df,0.4)
env.scenes = [scene]

new_trajectron = initialise_trajectron_calibrated_to_env(env)


### New Code Edits ###
# Needs to return a list of different means for each timetep and a list 
# of different stdev etc. 

## assume that ind refers to the particular timestep & horz refers to the prediction horizon
horz = 3
num_trajs = 3
mean_x, std_x, mean_y, std_y = predict_from_timestep(new_trajectron,horz,num_trajs,1,scene)
print(mean_x,std_x,mean_y,std_y)

print(scene)
for i in range(3):
    x_pos_list.append(1)
    y_pos_list.append(2)
    print(node_df)
