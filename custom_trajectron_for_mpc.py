import os
import statistics
import sys
import time
import json
import torch
#import dill
import random
import math
#import pathlib

# os.chdir("/home/kong35/Trajectron-plus-plus/trajectron/")
# print(os.getcwd())
sys.path.append(r"C:\Users\65932\OneDrive\Desktop\Gatech_Internship\Modified-Trajectron-\trajectron")

#import evaluation
import numpy as np
import visualization as vis
import pandas as pd
from argument_parser import args
from model.online.online_trajectron import OnlineTrajectron
from model.model_registrar import ModelRegistrar
from environment import Environment, Scene, Node, derivative_of
import matplotlib.pyplot as plt

### Code for generating an environment & scene based on custom x & y values for a particular pedestrian ###
    # Choose one of the model directory names under the experiment/*/models folders.
    # Possibilities are 'vel_ee', 'int_ee', 'int_ee_me', or 'robot'



if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = 'cpu'

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

#specifying the standardisation
standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

log_dir = r"C:\Users\65932\OneDrive\Desktop\Gatech_Internship\Modified-Trajectron-\experiments\nuScenes\models"
conf = "config.json"
model_dir = os.path.join(log_dir, 'int_ee_cassie')

# Load hyperparameters from json
config_file = os.path.join(model_dir, conf)
print(config_file)
if not os.path.exists(config_file):
    raise ValueError('Config json not found!')
with open(config_file, 'r') as conf_json:
    hyperparams = json.load(conf_json)
# Add hyperparams from arguments
hyperparams['dynamic_edges'] = args.dynamic_edges
hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
hyperparams['edge_addition_filter'] = args.edge_addition_filter
hyperparams['edge_removal_filter'] = args.edge_removal_filter
hyperparams['batch_size'] = args.batch_size
hyperparams['k_eval'] = args.k_eval
hyperparams['offline_scene_graph'] = args.offline_scene_graph
hyperparams['incl_robot_node'] = args.incl_robot_node
hyperparams['edge_encoding'] = not args.no_edge_encoding
hyperparams['use_map_encoding'] = args.map_encoding

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug

def obtain_node_df_from_txt(txt_filename,node_id):
    input_data_dict = dict()
    print('At', txt_filename)
    data = pd.read_csv(txt_filename, sep='\t', index_col=False, header=None)
    data.columns = ['frame_id', 'track_id', 'x', 'y']
    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

    data['frame_id'] = data['frame_id'] // 10

    data['frame_id'] -= data['frame_id'].min()

    data['type'] = env.NodeType.PEDESTRIAN
    data['node_id'] = data['track_id'].astype(str) #assigning each node id to a particular track id
    data.sort_values('frame_id', inplace=True)

    # Mean Position
    data['x'] = data['x'] - data['x'].mean()
    data['y'] = data['y'] - data['y'].mean()

    # max_timesteps = data['frame_id'].max()

    # Initialising the scene object (most relevant object that we use)

    # scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=augment if data_class == 'train' else None)
    print(data['node_id'])
    #for node_id in pd.unique(data['node_id']):
    #    print(node_id)
    node_df = data[data['node_id'] == node_id]
    assert np.all(np.diff(node_df['frame_id']) == 1)
    print(node_df)
    return node_df
    #     node_values = node_df[['pos_x', 'pos_y']].values

    #     if node_values.shape[0] < 2:
    #         continue

    #     new_first_idx = node_df['frame_id'].iloc[0]

    #     x = node_values[:, 0]
    #     y = node_values[:, 1]
    #     vx = derivative_of(x, scene.dt)
    #     vy = derivative_of(y, scene.dt)
    #     ax = derivative_of(vx, scene.dt)
    #     ay = derivative_of(vy, scene.dt)

    #     data_dict = {('position', 'x'): x,
    #                     ('position', 'y'): y,
    #                     ('velocity', 'x'): vx,
    #                     ('velocity', 'y'): vy,
    #                     ('acceleration', 'x'): ax,
    #                     ('acceleration', 'y'): ay}

    #     node_data = pd.DataFrame(data_dict, columns=data_columns)
    #     node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
    #     node.first_timestep = new_first_idx

    #     scene.nodes.append(node)
    # if data_class == 'train':
    #     scene.augmented = list()
    #     angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
    #     for angle in angles:
    #         scene.augmented.append(augment_scene(scene, angle))

    # print(scene)
    # scenes.append(scene)
    # print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

    # env.scenes = scenes

def generate_node(node_df,dt):
    start_time = time.clock()
    node_df.sort_values('frame_id', inplace=True)
    max_timesteps = node_df['frame_id'].max()

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name="test 1", aug_func=augment)

    node_frequency_multiplier = 1
   
    if node_df['x'].shape[0] < 2:
        print("Not feasible, only 1 position")
    

    if not np.all(np.diff(node_df['frame_id']) == 1):
        print('Occlusion')
         

    node_values = node_df[['x', 'y']].values
    x = node_values[:, 0]
    y = node_values[:, 1]
    vx = derivative_of(x, scene.dt)
    vy = derivative_of(y, scene.dt)
    ax = derivative_of(vx, scene.dt)
    ay = derivative_of(vy, scene.dt)

    data_dict = {('position', 'x'): x,
                    ('position', 'y'): y,
                    ('velocity', 'x'): vx,
                    ('velocity', 'y'): vy,
                    ('acceleration', 'x'): ax,
                    ('acceleration', 'y'): ay}
    print("original data dict is",data_dict)
    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian) 

    #create node_data corresponding to each timestep (likely 0.5 seconds)
    #data dict is an array which contains all the positions/ velocities/ 
    # accelerations of an object at its different timesteps
    node = Node(node_type=node_df.iloc[0]['type'], node_id="Pedestrian 1", data=node_data, frequency_multiplier=node_frequency_multiplier)
    node.first_timestep = node_df['frame_id'].iloc[0]  #assign each timestep
    print(time.clock()-start_time,"seconds")
    return node

def generate_node_from_pos_lsts(x_lst,y_lst,dt,node_type,node_id):
    start_time = time.clock()
    node_frequency_multiplier = 1
   
    if len(x_lst) < 2 or len(y_lst) < 2:
        print("Not feasible, only 1 position")     

    x = np.array(x_lst)
    print(x.shape)
    y = np.array(y_lst)
    

    vx = derivative_of(x, dt)
    print(vx.shape)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)

    data_dict = {('position', 'x'): x,
                    ('position', 'y'): y,
                    ('velocity', 'x'): vx,
                    ('velocity', 'y'): vy,
                    ('acceleration', 'x'): ax,
                    ('acceleration', 'y'): ay}
    print("new data dict is",data_dict)
    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian) 

    #create node_data corresponding to each timestep (likely 0.5 seconds)
    #data dict is an array which contains all the positions/ velocities/ 
    # accelerations of an object at its different timesteps
    node = Node(node_type=node_type, node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
    node.first_timestep = 0
    print(time.clock()-start_time,"seconds")
    return node

def initialise_pedestrian_scene(node_df,dt):
    """
    create a function that takes in a node_df (pandas dataframe which contains the different positions of a particular pedestrian)
    pandas dataframe with columns ['frame_id',
                                  'type',
                                  'node_id',
                                  'robot',
                                  'x', 'y', 'z',
                               'length',
                                  'width',
                                  'height',
                               'heading']), also takes in dt which is the value of each timestep
    """


    node_df.sort_values('frame_id', inplace=True)
    max_timesteps = node_df['frame_id'].max()

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name="test 1", aug_func=augment)

    node_frequency_multiplier = 1
   
    if node_df['x'].shape[0] < 2:
        print("Not feasible, only 1 position")
    

    if not np.all(np.diff(node_df['frame_id']) == 1):
        print('Occlusion')
         

    node_values = node_df[['x', 'y']].values
    x = node_values[:, 0]
    y = node_values[:, 1]
    

    vx = derivative_of(x, scene.dt)
    vy = derivative_of(y, scene.dt)
    ax = derivative_of(vx, scene.dt)
    ay = derivative_of(vy, scene.dt)

    data_dict = {('position', 'x'): x,
                    ('position', 'y'): y,
                    ('velocity', 'x'): vx,
                    ('velocity', 'y'): vy,
                    ('acceleration', 'x'): ax,
                    ('acceleration', 'y'): ay}
    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian) 

    #create node_data corresponding to each timestep (likely 0.5 seconds)
    #data dict is an array which contains all the positions/ velocities/ 
    # accelerations of an object at its different timesteps
    node = Node(node_type=node_df.iloc[0]['type'], node_id="Pedestrian 1", data=node_data, frequency_multiplier=node_frequency_multiplier)
    node.first_timestep = node_df['frame_id'].iloc[0]  #assign each timestep
    # if node_df.iloc[0]['robot'] == True:
    #     node.is_robot = True
    #     scene.robot = node
    scene.nodes.append(node)
    return scene

def initialise_dummy_scene(max_timesteps,dt):
    scene = Scene(timesteps=max_timesteps, dt=dt, name="test scene", aug_func=augment)
    return scene

def generate_pedestrian_obstacle_for_trajectron(xs,ys):
        #x_pos_list = [i for i in range(0,20)]
        #y_pos_list = [i for i in range(0,20)]
        #rng = np.random.RandomState()
        #xs0 = (np.arange(x0, xf, inc)).reshape(-1,1)
        #ys0 = (np.arange(y0, yf, inc)).reshape(-1,1)

        #xs = xs0 + (0.01 * rng.standard_normal(size=len(xs0))).reshape(-1,1)
        #xs = xs0 #+ (np.random.uniform(-0.2,0.2,size=len(xs0))).reshape(-1,1)
        #ys0 = xs*0 + y0 
        #ys = ys0 + (0.02 * rng.standard_normal(size=len(xs0))).reshape(-1,1)
        #ys = ys0 #+ (np.random.uniform(-0.2,0.2,size=len(xs0))).reshape(-1,1)
        x_pos_lst = xs[:,0].tolist()
        y_pos_lst = ys[:,0].tolist()
        #print(x_pos_lst)
        #print(y_pos_lst)
        frame_id = [i for i in range(0,len(x_pos_lst))]
        node_df = pd.DataFrame({'frame_id': frame_id,
                                            'type': env.NodeType.PEDESTRIAN,
                                            #'node_id': annotation['instance_token'],
                                            'robot': False,
                                            'x': x_pos_lst,
                                            'y': y_pos_lst,
                                            'z': 0,
                                            'length': 0.4,
                                            'width': 0.5,
                                            'height': 0.3,
                                            'heading': 0})
        return node_df

def create_online_env(env, hyperparams, scene_idx, init_timestep):
    test_scene = env.scenes[scene_idx]

    online_scene = Scene(timesteps=init_timestep + 1,
                         map=test_scene.map,
                         dt=test_scene.dt)
    online_scene.nodes = test_scene.get_nodes_clipped_at_time(
        timesteps=np.arange(init_timestep - hyperparams['maximum_history_length'],
                            init_timestep + 1),
        state=hyperparams['state'])
    online_scene.robot = test_scene.robot
    online_scene.calculate_scene_graph(attention_radius=env.attention_radius,
                                       edge_addition_filter=hyperparams['edge_addition_filter'],
                                       edge_removal_filter=hyperparams['edge_removal_filter'])

    return Environment(node_type_list=env.node_type_list,
                       standardization=env.standardization,
                       scenes=[online_scene],
                       attention_radius=env.attention_radius,
                       robot_type=env.robot_type)

def initialise_trajectron_calibrated_to_env(env):

    # Creating a dummy environment with a single scene that contains information about the world.
    # When using this code, feel free to use whichever scene index or initial timestep you wish.
    scene_idx = 0

    # You need to have at least acceleration, so you want 2 timesteps of prior data, e.g. [0, 1],
    # so that you can immediately start incremental inference from the 3rd timestep onwards.
    init_timestep = 1

    # creating a singular environemnt from a particular scene within 
    # collection of scenes within the env datasets
    online_env = create_online_env(env, hyperparams, scene_idx, init_timestep)

    #combine & load model
    model_registrar = ModelRegistrar(model_dir, args.eval_device)
    model_registrar.load_models(iter_num=12)

    trajectron = OnlineTrajectron(model_registrar,
                                  hyperparams,
                                  args.eval_device)

   
    # Here's how you'd incrementally run the model, e.g. with streaming data; setting the environment
    # accordingly for the trajectron to make predictions 
    trajectron.set_environment(online_env, init_timestep)
    return trajectron

def predict_from_timestep(trajectron,prediction_horizon,num_trajectories,timestep,scene):
    predictions_at_timestep=dict()
    input_dict = scene.get_clipped_input_dict(timestep, hyperparams['state']) #get the different pedestrian positions & details of the scene 
    #TODO: => new project: to change online, just change input_dict to the streams of data coming in instead of having it intialised from the scene
    maps = None
    start = time.time()
    ##@TODO: Try the trajectron prediction of entire distribution
    #print("the input dict is",input_dict)
    dists, preds = trajectron.incremental_forward(input_dict,
                                                    maps,
                                                    prediction_horizon=prediction_horizon, #how long into the future do you want to predict?
                                                    num_samples=num_trajectories, #how many different trajectories do you want?
                                                    robot_present_and_future=None,
                                                    full_dist=True)
    end = time.time()
    #print("t=%d: took %.2f s (= %.2f Hz) w/ %d nodes and %d edges" % (timestep, end - start,
    #                                                                    1. / (end - start), len(trajectron.nodes),
    #                                                                    trajectron.scene_graph.get_num_edges()))
    for node in scene.nodes:
        if node in preds:
            raw_predictions = preds[node] 
    for p_horz in range(raw_predictions.shape[2]): 
        new_key = "t" + str(timestep+1+p_horz)
        all_x_at_timestep = []
        all_y_at_timestep = []
        for traj_n in range(raw_predictions.shape[1]):
            all_x_at_timestep.append(raw_predictions[0,traj_n,p_horz,0]) #obtains x value at each timestep
            all_y_at_timestep.append(raw_predictions[0,traj_n,p_horz,1]) #obtains y value at each timestep
        predictions_at_timestep[new_key] = [all_x_at_timestep,all_y_at_timestep]
    mean_stdev_info = convert_traj_to_mean_stdev_lst(predictions_at_timestep)
    return mean_stdev_info
    

def convert_traj_to_mean_stdev_dict(input_dict):
    dict_with_mean_stdev = dict()
    for timestep in input_dict:
        dict_with_mean_stdev[timestep]=dict()
        predictions = input_dict[timestep]
        x_vals = list(np.float_(predictions[0]))
        y_vals = list(np.float_(predictions[1]))
        dict_with_mean_stdev[timestep] = [[statistics.mean(x_vals),statistics.stdev(x_vals)],[statistics.mean(y_vals),statistics.stdev(y_vals)]]
    return dict_with_mean_stdev

def convert_traj_to_mean_stdev_lst(input_dict):
    all_mean_x = []
    all_std_x = []
    all_mean_y = []
    all_std_y = []

    for timestep in input_dict:
        predictions = input_dict[timestep]
        x_vals = list(np.float_(predictions[0]))
        y_vals = list(np.float_(predictions[1]))
        all_mean_x.append(statistics.mean(x_vals))
        all_std_x.append(statistics.stdev(x_vals))
        all_mean_y.append(statistics.mean(y_vals))
        all_std_y.append(statistics.stdev(y_vals))
    return all_mean_x, all_std_x, all_mean_y, all_std_y

def visualise_dist_of_samples(input_dict):
    for timestep in input_dict:
        for pred_time in input_dict[timestep]:
            x_values = input_dict[timestep][pred_time][0]
            y_values = input_dict[timestep][pred_time][1]
            plt.figure(figsize=(16,12))
            plt.subplot(2,1,1)
            n_bins = int(math.ceil(math.sqrt(len(x_values))))
            print(n_bins)
            plt.hist(x_values, bins=n_bins)
            plt.subplot(2,1,2)
            plt.hist(y_values,bins=n_bins)
            plt.show()
            break
        break


if __name__ == '__main__':
     #generate_pedestrian_obstacle_for_trajectron(0,10,0,10,1)
     obtain_node_df_from_txt("/home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/students001_val.txt",'142')
