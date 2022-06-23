#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
#from numpy.random import default_rng
from casadi import *
from casadi.tools import *
#from CVaR_valueIteration import CVaR_space
from GP import *
from obstacle_traj import *
from MDP_next_waypoint import *
import pdb
import sys
import time
sys.path.append('../../')
import do_mpc
from custom_trajectron_for_mpc import *
from statistics import mean,stdev


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 3,
        't_step': 1,
        'store_full_solution':True,
        #'nl_cons_check_colloc_points': True,
    }

    mpc.set_param(**setup_mpc)

    #xg=0
    #yg=6
    xdg=0
    ydg=0

   
    mterm = (-(model.x['x',1] - xdg)**2 + (model.x['x',3] - ydg)**2) + 10*((model.x['x',0] - model.tvp['xg'])**2 + (model.x['x',2] - model.tvp['yg'])**2)  #model.aux['cost']
    lterm = (-(model.x['x',1] - xdg)**2 + (model.x['x',3] - ydg)**2) + 10*((model.x['x',0] - model.tvp['xg'])**2 + (model.x['x',2] - model.tvp['yg'])**2)  #+ 10* model.aux['obstacle_distance']**2#model.aux['cost'] # terminal cost
   # mterm = ((model.x['x',1] - 1)*(model.x['x',1] - 1)+ ((model.x['x',3] - 0)*(model.x['x',3] - 0)) + (model.x['x',2] - 0)*(model.x['x',2] - 0)) #model.aux['cost']
  #  lterm = ((model.x['x',1] - 1)*(model.x['x',1] - 1)+ ((model.x['x',3] - 0)*(model.x['x',3] - 0)) + (model.x['x',2] - 0)*(model.x['x',2] - 0)) #model.aux['cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)

    max_x = np.array([[100.0], [1], [100.0], [1]])
    min_x = np.array([[-10.0], [-1], [-10.0], [-1]])
    max_u = np.array([[0.3], [0.3]])
    min_u = np.array([[-0.3], [-0.3]])

    mpc.bounds['lower','_x','x'] =  min_x
    mpc.bounds['upper','_x','x'] =  max_x

   # mpc.bounds['lower','_u','u'] =  min_u
   # mpc.bounds['upper','_u','u'] =  max_u


    ### New Code for generating dynamic obstacle ###
    # starting_x = 0
    # final_x = 10
    # starting_y = 7
    # final_y = 1
    # inc = 0.35

    #xs, ys, d_obs_x, d_obs_y, X = obstacle_obsrv(setup_mpc['t_step'], starting_x, final_x, starting_y, final_y, inc)
    # node_df = generate_pedestrian_obstacle_for_trajectron(xs,ys)
    # 42
    # 236
    # 343
    # 232
    # 233
    # 235
    # 234
    # 344
    # 237
    # 346
    # 347
    # 149
    # 238
    # 150
    # 239

    ## initialise node_df from eth txt file
    node_df = obtain_node_df_from_txt("/home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/students001_val.txt",'142')
    xs = node_df['x'].to_numpy().reshape(-1,1) 
    ys = node_df['y'].to_numpy().reshape(-1,1)
    
    ## initialise scene from node_df 
    time_per_timestep = 0.4
    scene = initialise_pedestrian_scene(node_df,time_per_timestep)     ## specify the amount of time for each timestep                               
    env.scenes=[scene]
    new_trajectron = initialise_trajectron_calibrated_to_env(env)


    #####
    #xs, ys, d_obs_x, d_obs_y, X = obstacle_obsrv(setup_mpc['t_step'], 0, 8, 1.2, 0, 0.35)
    #xs2, ys2, d_obs_x2, d_obs_y2, X2 = obstacle_obsrv(setup_mpc['t_step'], -3, 2, 4, 0, 0.2)
    
    last =len(xs)-1
    s=[]
    s.append(40)
    x_global=[]
    x_global.append(0.5)
    y_global=[]
    y_global.append(0.5)

    tvp_template = mpc.get_tvp_template()
    
    def tvp_fun(t_ind):
    
        ind = t_ind // setup_mpc['t_step']
        int_ind = int(ind)
        # mem=3

        xcurr = mpc.data['_x','x',0]
        ycurr = mpc.data['_x','x',2]
        if int_ind < last:
            tvp_template['_tvp',:, 'dyn_obs_y'] = ys[int_ind]
            tvp_template['_tvp',:, 'dyn_obs_x'] = xs[int_ind]
            '''
            tvp_template['_tvp',:, 'dyn_obs_y2'] = ys2[int_ind]
            tvp_template['_tvp',:, 'dyn_obs_x2'] = xs2[int_ind]
            '''
            #tvp_template['_tvp',:, 'D'] = 0.05 
            if ind > 1:
                # Keane: Most relevant parts > replace GP by taking mean/ std of x & y
                # and deploying them accordingly aka output from trajectron needs to be x & y 
                # these are delta x and y values, which is why you have to add them to current position to get new position of x and y
                conf = 1.96
                horz = 1
                num_trajs = 5

                ### New Code Edits ###
                # Needs to return a list of different means for each timetep and a list 
                # of different stdev etc. 

                ## assume that ind refers to the particular timestep & horz refers to the prediction horizon
                mean_x, std_x, mean_y, std_y = predict_from_timestep(new_trajectron,horz,num_trajs,int_ind,scene)
                
                ##########
                
                # mean_x, std_x, mean_y, std_y = GP(d_obs_x[int_ind-mem:int_ind].reshape(-1,1), d_obs_y[int_ind-mem:int_ind].reshape(-1,1), X[int_ind-mem:int_ind], X, int_ind)
                

                # Keane: Plan > you can obtain a sample of trajectories and then obtain the mean and standard deviation from them, this
                # is to obtain CVaR values, as well as the particular confidence level so that we can draw a radius around the dynamic object
                # radius is drawn/ configured by h and w values 
                # On the other hand, the sample of trajectories can give us a deterministic position that we can key into the tvp_template below    
            
                #delta_x, delta_y, h, w = pred(mean_x, std_x, mean_y, std_y, conf, horz) # obtaining deterministic x & y values from mean x & y

                # new_x, new_y, h, w = pred(mean_x, std_x, mean_y, std_y, conf, horz)

                # tvp_template['_tvp',:, 'dyn_obs_y_pred'] = ys[int_ind] + delta_y #new y to be inputted (instead of delta y or mean y)
                # tvp_template['_tvp',:, 'dyn_obs_x_pred'] = xs[int_ind] + delta_x    #new x to be inputted (instead of delta x or mean x)
                tvp_template['_tvp',:, 'dyn_obs_y_pred'] = mean_y[0]
                tvp_template['_tvp',:, 'dyn_obs_x_pred'] = mean_x[0]

                tvp_template['_tvp',:, 'dyn_obs_ry'] = 1 + std_y[0]*conf
                tvp_template['_tvp',:, 'dyn_obs_rx'] = 1 + std_x[0]*conf
                '''
                beta = 0.95
                CVaR, CVaR_space = CVaR_map_mpc(beta, mean_x[0], mean_y[0], std_x[0], std_y[0], 8, 6, xs[int_ind], ys[int_ind])
                #tvp_template['_tvp',:,'cvar_space']=CVaR_space
                s_c, sx, sy = eucl2disc(xcurr[int_ind-1],ycurr[int_ind-1],8,6,1)
                w_x, w_y, s_n = next_waypoint(s_c, CVaR, 1, 8, 6)

                x_global.append(w_x)
                y_global.append(w_y)
                x_goal = sum(x_global)
                y_goal = sum(y_global)
                
                tvp_template['_tvp',:, 'xg'] =  sx*1 + w_x - 0.5
                tvp_template['_tvp',:, 'yg'] =  sy*1 + w_y - 0.5
                '''


            

                '''
                mean_x2, std_x2, mean_y2, std_y2 = GP(d_obs_x2[int_ind-mem:int_ind].reshape(-1,1), d_obs_y2[int_ind-mem:int_ind].reshape(-1,1), X2[int_ind-mem:int_ind], X2, int_ind)
                conf = 1.96
                horz = 3
                delta_x2, delta_y2, h2, w2 = pred(mean_x2, std_x2, mean_y2, std_y2, conf, horz)
                tvp_template['_tvp',:, 'dyn_obs_y_pred2'] = ys2[int_ind] + delta_y2
                tvp_template['_tvp',:, 'dyn_obs_x_pred2'] = xs2[int_ind] + delta_x2
                tvp_template['_tvp',:, 'dyn_obs_ry2'] = 1 + h2
                tvp_template['_tvp',:, 'dyn_obs_rx2'] = 1 + w2
                '''

                #D = sqrt(((delta_x_2))**2 + ((delta_y_2))**2)-1
                #tvp_template['_tvp',:, 'D'] = D 

            else:
                '''
                tvp_template['_tvp',:, 'dyn_obs_y_pred'] = ys[int_ind]
                tvp_template['_tvp',:, 'dyn_obs_x_pred'] = xs[int_ind]
                tvp_template['_tvp',:, 'dyn_obs_ry'] = 1
                tvp_template['_tvp',:, 'dyn_obs_rx'] = 1

                
                tvp_template['_tvp',:, 'dyn_obs_y_pred2'] = ys2[int_ind]
                tvp_template['_tvp',:, 'dyn_obs_x_pred2'] = xs2[int_ind]
                tvp_template['_tvp',:, 'dyn_obs_ry2'] = 1
                tvp_template['_tvp',:, 'dyn_obs_rx2'] = 1
                #tvp_template['_tvp',:, 'D'] = 0.05 
                '''
                tvp_template['_tvp',:, 'xg'] = 4.5
                tvp_template['_tvp',:, 'yg'] = -2
               
        else:
            tvp_template['_tvp',:, 'dyn_obs_y_pred'] = ys[last]
            tvp_template['_tvp',:, 'dyn_obs_x_pred'] = xs[last]
            tvp_template['_tvp',:, 'dyn_obs_ry'] = 1
            tvp_template['_tvp',:, 'dyn_obs_rx'] = 1
            tvp_template['_tvp',:, 'dyn_obs_y'] = ys[last]
            tvp_template['_tvp',:, 'dyn_obs_x'] = xs[last]
            
            '''
            tvp_template['_tvp',:, 'dyn_obs_y_pred2'] = ys2[last]
            tvp_template['_tvp',:, 'dyn_obs_x_pred2'] = xs2[last]
            tvp_template['_tvp',:, 'dyn_obs_ry2'] = 1
            tvp_template['_tvp',:, 'dyn_obs_rx2'] = 1
            tvp_template['_tvp',:, 'dyn_obs_y2'] = ys2[last]
            tvp_template['_tvp',:, 'dyn_obs_x2'] = xs2[last]
            
            '''
            tvp_template['_tvp',:, 'xg'] = 4.5
            tvp_template['_tvp',:, 'yg'] = -2

            
        #dist = sqrt(((model.x['x',0] - model.tvp['xg'])**2 + (model.x['x',2] - model.tvp['yg'])**2))

        #if ind <= 10:
        
        #else:
         #       tvp_template['_tvp',:, 'xg'] = 0
          #      tvp_template['_tvp',:, 'yg'] = 6



        #tvp_template['_tvp',:, 'dyn_obs'] = 8.8 - ind*(0.225)
        #rng = default_rng()
        #vals = rng.standard_normal()
        #tvp_template['_tvp',:, 'dyn_obs_y'] = 8.8 - ind*(0.225+np.abs(vals)*0.3)
        #tvp_template['_tvp',:, 'dyn_obs_x'] = 8 + np.abs(vals)*0.1



        tvp_template['_tvp',:, 'stance'] = (-1)**(ind)

        

        
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    
    

    #if tvp_template['_tvp',-1, 'stance'] > 0:
       # mpc.bounds['lower','_u','u'] =  np.array([[0.0], [-0.5]])
      #  mpc.bounds['upper','_u','u'] =  np.array([[0.4], [-0.2model.aux['obstacle_distance']
    #mpc.bounds['lower','_u','u'] =  min_u
    #mpc.bounds['upper','_u','u'] =  max_u

    omega=sqrt(9.81/0.8)
    
    mpc.set_nl_cons('obstacles', model.aux['obstacle_distance'], 0)#, penalty_term_cons=1e1)
    #mpc.set_nl_cons('obstacles', (1-0.5)*model.aux['hk1']- model.aux['hk1_n'],0)
    #mpc.set_nl_cons('dyn_obstacles', model.aux['dyn_obstacle_distance'], 0)
    mpc.set_nl_cons('grizzle1', model.aux['grizzle1'], 0)
    mpc.set_nl_cons('grizzle2', model.aux['grizzle2'], 0)
    mpc.set_nl_cons('grizzle11', model.aux['grizzle11'], 0)
    mpc.set_nl_cons('grizzle22', model.aux['grizzle22'], 0)
    #mpc.set_nl_cons('grizzle3', -model.aux['grizzle3'], 0)


    #mpc.set_nl_cons('PSP', -model.aux['psp_safety'], 0)
    #mpc.set_nl_cons('kin', model.aux['kin_safety'], 0)
    #mpc.set_nl_cons('kin_min', model.aux['kin_safety_min'], 0)

    
    #print(model.u['u',1])
    #mpc.set_nl_cons('PSP2', -fabs(model.u['u',1]), ub=-0.05)



    mpc.setup()

    return mpc