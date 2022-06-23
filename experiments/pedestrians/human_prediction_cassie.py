import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron_edited import Trajectron
from environment import Environment, Scene, Node
from utils import maybe_makedirs
from environment import derivative_of, Environment, Scene
import evaluation
import visualization as vis

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


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    #model_dir refers to the directory of the model that you select to run with
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env) #set the scene for trajectron to predict
    trajectron.set_annealing_params()
    return trajectron, hyperparams

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
    
    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint) #load the particular model to run it

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])

        ############### MOST LIKELY ###############
        ## Outputing deterministic predictions 
        # print("Predicting Most Likely")
        # for i, scene in enumerate(scenes):
        #     print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
        #     timesteps = np.arange(scene.timesteps)

        #     predictions = eval_stg.predict(scene,
        #                                    timesteps,
        #                                    ph,
        #                                    num_samples=1,
        #                                    min_history_timesteps=7,
        #                                    min_future_timesteps=12,
        #                                    z_mode=False,
        #                                    gmm_mode=True,
        #                                    full_dist=True)  # This will trigger grid sampling


        ############### MODE Z ###############
        # print("Predicting Mode Z")
        # for i, scene in enumerate(scenes):
        #     print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
        #     for t in tqdm(range(0, scene.timesteps, 10)):
        #         timesteps = np.arange(t, t + 10)
        #         predictions = eval_stg.predict(scene,
        #                                        timesteps,
        #                                        ph,
        #                                        num_samples=2000,
        #                                        min_history_timesteps=7,
        #                                        min_future_timesteps=12,
        #                                        z_mode=True,
        #                                        full_dist=False)

        #         if not predictions:
        #             continue


        ############### BEST OF 20 ###############
        # print("-- Predicting best of 20")
        # for i, scene in enumerate(scenes):
        #     print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
        #     for t in tqdm(range(0, scene.timesteps, 10)):
        #         timesteps = np.arange(t, t + 10)
        #         predictions = eval_stg.predict(scene,
        #                                        timesteps,
        #                                        ph,
        #                                        num_samples=20,
        #                                        min_history_timesteps=7,
        #                                        min_future_timesteps=12,
        #                                        z_mode=False,
        #                                        gmm_mode=False,
        #                                        full_dist=False)

        #         if not predictions:
        #             continue


        ############### FULL ###############
        # Probablistic full sequence, which is what we want
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Predicting Full")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=2000,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)

                if not predictions:
                    continue
                print(predictions)
        # structure for predictions appears to be in the form of {ts:{node:[array of prediction info]}}   
                break
        
        #@TODO: Change the outputs to obtain a visualisation/ probability distribution
        #         batch_error_dict = evaluation.compute_batch_statistics(predictions,
        #                                                                scene.dt,
        #                                                                max_hl=max_hl,
        #                                                                ph=ph,
        #                                                                node_type_enum=env.NodeType,
        #                                                                map=None,
        #                                                                prune_ph_to_future=True)
        #         eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
        #         eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
        #         eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        # pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_full.csv'))
        # pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_full.csv'))
        # pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_full.csv'))

## Usage of online running trajectories in order to ensure accuracy (look at test_online.py); this also allows you to output a particular trajectory

# model_path: /home/kong35/Trajectron-plus-plus/experiments/pedestrians/models/cassie_eth_model_ar3
# possible command:
# python human_prediction_cassie.py --model /home/kong35/Trajectron-plus-plus/experiments/pedestrians/models/cassie_eth_model_ar3 --checkpoint 3 --data /home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/students001_val.txt --output_path results --output_tag cassie_trial_A_12 --node_type PEDESTRIAN
# python human_prediction_cassie.py --model /home/kong35/Trajectron-plus-plus/experiments/pedestrians/models/cassie_eth_model_ar3 ----data /home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/students001_val.txt --map_encoding --incl_robot_node