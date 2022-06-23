import sys
import os
import dill
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron_edited import Trajectron
from model.online.online_trajectron import OnlineTrajectron
from environment import Environment, Scene, Node
from utils import maybe_makedirs
from environment import derivative_of, Environment, Scene
import evaluation
import visualization as vis
from argument_parser import args

####################################
###Code for manipulating txt data###
####################################

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

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
    }
}

def get_maps_for_input(input_dict, scene, hyperparams):
    scene_maps = list()
    scene_pts = list()
    heading_angles = list()
    patch_sizes = list()
    nodes_with_maps = list()
    for node in input_dict:
        if node.type in hyperparams['map_encoder']:
            x = input_dict[node]
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = hyperparams['map_encoder'][node.type]['patch_size']

            scene_maps.append(scene_map)
            scene_pts.append(map_point)
            heading_angles.append(heading_angle)
            patch_sizes.append(patch_size)
            nodes_with_maps.append(node)

    if heading_angles[0] is None:
        heading_angles = None
    else:
        heading_angles = torch.Tensor(heading_angles)

    maps = scene_maps[0].get_cropped_maps_from_scene_map_batch(scene_maps,
                                                               scene_pts=torch.Tensor(scene_pts),
                                                               patch_size=patch_sizes[0],
                                                               rotation=heading_angles)

    maps_dict = {node: maps[[i]] for i, node in enumerate(nodes_with_maps)}
    return maps_dict

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

#making a pandas multiIndex
    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

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

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0
maybe_makedirs('../processed')
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

####################################
###Code for running prediction##
####################################


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


def predict_and_visualise(env):
    model_dir = args.model
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
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

    if env.robot_type is None and hyperparams['incl_robot_node']:
        env.robot_type = env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in env.scenes:
            scene.add_robot_from_nodes(env.robot_type)

    scene_idx = 0
    init_timestep = 1
    # You need to have at least acceleration, so you want 2 timesteps of prior data, e.g. [0, 1],
    # so that you can immediately start incremental inference from the 3rd timestep onwards.    init_timestep = 1
    
    eval_scene = env.scenes[scene_idx]

    #creating an online environment from a particular scene
    online_scene = Scene(timesteps=init_timestep + 1,
                         map=eval_scene.map,
                         dt=eval_scene.dt)
    online_scene.nodes = eval_scene.get_nodes_clipped_at_time(
        timesteps=np.arange(init_timestep - hyperparams['maximum_history_length'],
                            init_timestep + 1),
        state=hyperparams['state'])
    online_scene.robot = eval_scene.robot
    online_scene.calculate_scene_graph(attention_radius=env.attention_radius,
                                       edge_addition_filter=hyperparams['edge_addition_filter'],
                                       edge_removal_filter=hyperparams['edge_removal_filter'])

    one_scene_env = Environment(node_type_list=env.node_type_list,
                       standardization=env.standardization,
                       scenes=[online_scene],
                       attention_radius=env.attention_radius,
                       robot_type=env.robot_type)

    
    #load the model according to the online environment
    # this will automatically set the model env to that one scene
    traj, hyperparams = load_model(model_dir, one_scene_env, hyperparams, ts=100)
    
    #for each timestep in the same scene, obtain the predictions
    for timestep in range(init_timestep + 1, eval_scene.timesteps):
        input_dict = eval_scene.get_clipped_input_dict(timestep, hyperparams['state'])

        maps = None
        if hyperparams['use_map_encoding']:
            #configure to use maps
            maps = get_maps_for_input(input_dict, eval_scene, hyperparams)

        robot_present_and_future = None

        #for each time step, add the trajectory of the robot in the form of an array
        if eval_scene.robot is not None and hyperparams['incl_robot_node']:
            robot_present_and_future = eval_scene.robot.get(np.array([timestep,
                                                                      timestep + hyperparams['prediction_horizon']]),
                                                            hyperparams['state'][eval_scene.robot.type],
                                                            padding=0.0)
            robot_present_and_future = np.stack([robot_present_and_future, robot_present_and_future], axis=0)
            # robot_present_and_future += adjustment

        start = time.time()

        #obtain the predictions from trajectron in terms of the distribution and predictions
        #takes in a particular input_dictionary, which refers to a particular scene
        dists, preds = traj.incremental_forward(input_dict,
                                                      maps,
                                                      prediction_horizon=6,
                                                      num_samples=1,
                                                      robot_present_and_future=robot_present_and_future,
                                                      full_dist=True)
        end = time.time()
        print("t=%d: took %.2f s (= %.2f Hz) w/ %d nodes and %d edges" % (timestep, end - start,
                                                                          1. / (end - start), len(trajectron.nodes),
                                                                          trajectron.scene_graph.get_num_edges()))

        detailed_preds_dict = dict()
        for node in eval_scene.nodes:
            if node in preds:
                detailed_preds_dict[node] = preds[node]

        fig, ax = plt.subplots()
        vis.visualize_distribution(ax,
                                   dists)
        vis.visualize_prediction(ax,
                                 {timestep: preds},
                                 eval_scene.dt,
                                 hyperparams['maximum_history_length'],
                                 hyperparams['prediction_horizon'])

        if eval_scene.robot is not None and hyperparams['incl_robot_node']:
            robot_for_plotting = eval_scene.robot.get(np.array([timestep,
                                                                timestep + hyperparams['prediction_horizon']]),
                                                      hyperparams['state'][eval_scene.robot.type])
            # robot_for_plotting += adjustment

            ax.plot(robot_for_plotting[1:, 1], robot_for_plotting[1:, 0],
                    color='r',
                    linewidth=1.0, alpha=1.0)

            # Current Node Position
            circle = plt.Circle((robot_for_plotting[0, 1],
                                 robot_for_plotting[0, 0]),
                                0.3,
                                facecolor='r',
                                edgecolor='k',
                                lw=0.5,
                                zorder=3)
            ax.add_artist(circle)

        fig.savefig(os.path.join(model_dir,'pred_figs', f'pred_{timestep}.pdf'), dpi=300)
        plt.close(fig)


def load_model(model_dir, env, hyper, ts=100):
    #model_dir refers to the directory of the model that you select to run with
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    trajectron = OnlineTrajectron(model_registrar, hyper,'cpu')

    trajectron.set_environment(env) #set the scene for trajectron to predict
    trajectron.set_annealing_params()
    return trajectron, hyper

def generate_scenes_from_txt(txt_dir):
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []
    
    print('At',txt_dir)
    input_data_dict = dict()

    data = pd.read_csv(txt_dir, sep='\t', index_col=False, header=None)
    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

    data['frame_id'] = data['frame_id'] // 10

    data['frame_id'] -= data['frame_id'].min()

    data['node_type'] = 'PEDESTRIAN'
    data['node_id'] = data['track_id'].astype(str)
    data.sort_values('frame_id', inplace=True)

    # Mean Position
    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

    max_timesteps = data['frame_id'].max()

    # Initialising the scene object (most relevant object that we use)

    scene = Scene(timesteps=max_timesteps+1, dt=dt, name= "scene_trial_one", aug_func=None)

    for node_id in pd.unique(data['node_id']):

        node_df = data[data['node_id'] == node_id]
        assert np.all(np.diff(node_df['frame_id']) == 1)

        node_values = node_df[['pos_x', 'pos_y']].values

        if node_values.shape[0] < 2:
            continue

        new_first_idx = node_df['frame_id'].iloc[0]

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

        node_data = pd.DataFrame(data_dict, columns=data_columns)
        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
        node.first_timestep = new_first_idx

        scene.nodes.append(node)
    # if data_class == 'train':
    #     scene.augmented = list()
    #     angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
    #     for angle in angles:
    #         scene.augmented.append(augment_scene(scene, angle))

    print(scene)

    #after generating the scene, appending all the scenes to the data struct
    #environment can contain scenes which contains many scenes of diff traj
    scenes.append(scene)
    env.scenes = scenes
    return env



if __name__ == "__main__":

#### Code for creating scene objects from txt files for Trajectron
    

    #TODO: possibly remove the dill parts below, as we are just dumping env

    print(f"Linear: {l}")
    print(f"Non-Linear: {nl}")

    ## Loading the trajectron model

    #obtain the environment from txt file directly
    env = generate_scenes_from_txt(args.data)
    
    predict_and_visualise(env)

# model_path: /home/kong35/Trajectron-plus-plus/experiments/pedestrians/models/cassie_eth_model_ar3
# possible command:
# python human_prediction_cassie.py --model /home/kong35/Trajectron-plus-plus/experiments/pedestrians/models/cassie_eth_model_ar3 --checkpoint 3 --data /home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/students001_val.txt --output_path results --output_tag cassie_trial_A_12 --node_type PEDESTRIAN
# python human_prediction_visualise_cassie.py --model /home/kong35/Trajectron-plus-plus/experiments/pedestrians/models/cassie_eth_model_ar3 --data /home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/students001_val.txt --map_encoding --incl_robot_node