import os
import time
import json
import torch
import dill
import random
import pathlib
import evaluation
import numpy as np
import visualization as vis
import pandas as pd
from argument_parser import args
from model.online.online_trajectron import OnlineTrajectron
from model.model_registrar import ModelRegistrar
from environment import Environment, Scene, Node, derivative_of
import matplotlib.pyplot as plt

### Code for generating an environment & scene based on custom x & y values for a particular pedestrian ###

#specifying the standardisation

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug

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

def generate_env_with_scene_from_dataframe(node_df,dt):
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
                               'heading'])
    """


    # scene_id = int(ns_scene['name'].replace('scene-', ''))
    ### Commented out code which obtains the positions of different objects from NuScenes dataset directly
    #  data = pd.DataFrame(columns=['frame_id',
    #                              'type',
    #                              'node_id',
    #                              'robot',
    #                              'x', 'y', 'z',
    #                              'length',
    #                              'width',
    #                              'height',
    #                              'heading'])
    # # Obtain sample token from ns_scene input, which will be used to 
    # # obtain the sample scene 
    # sample_token = ns_scene['first_sample_token']

    # #use sample token to obtain sample from nusc input (full dataset)
    # sample = nusc.get('sample', sample_token)
    # frame_id = 0
    # while sample['next']:
    #     annotation_tokens = sample['anns']
    #     for annotation_token in annotation_tokens:
    #         annotation = nusc.get('sample_annotation', annotation_token)
    #         category = annotation['category_name']
    #         if len(annotation['attribute_tokens']):
    #             attribute = nusc.get('attribute', annotation['attribute_tokens'][0])['name']
    #         else:
    #             continue

    #         if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
    #             our_category = env.NodeType.PEDESTRIAN
    #         elif 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
    #             our_category = env.NodeType.VEHICLE
    #         else:
    #             continue

    #         #building a data array of the different objects that we will 
    #         #subsequently analyse and compute heading, vel, position, acc
    #         #etc, 
    #         data_point = pd.Series({'frame_id': frame_id,
    #                                 'type': our_category,
    #                                 'node_id': annotation['instance_token'],
    #                                 'robot': False,
    #                                 'x': annotation['translation'][0],
    #                                 'y': annotation['translation'][1],
    #                                 'z': annotation['translation'][2],
    #                                 'length': annotation['size'][0],
    #                                 'width': annotation['size'][1],
    #                                 'height': annotation['size'][2],
    #                                 'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0]})
    #         data = data.append(data_point, ignore_index=True)

    #     # Ego Vehicle
    #     our_category = env.NodeType.VEHICLE
    #     sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    #     annotation = nusc.get('ego_pose', sample_data['ego_pose_token'])
    #     data_point = pd.Series({'frame_id': frame_id,
    #                             'type': our_category,
    #                             'node_id': 'ego',
    #                             'robot': True,
    #                             'x': annotation['translation'][0],
    #                             'y': annotation['translation'][1],
    #                             'z': annotation['translation'][2],
    #                             'length': 4,
    #                             'width': 1.7,
    #                             'height': 1.5,
    #                             'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
    #                             'orientation': None})
    #     data = data.append(data_point, ignore_index=True)

    #     sample = nusc.get('sample', sample['next'])
    #     frame_id += 1
    #     ## Each frame id corresponds to each timestamp which contains the different objects and their x,y,z positions 

    # if len(data.index) == 0:
    #     return None
    #print(node_df)
    node_df.sort_values('frame_id', inplace=True)
    max_timesteps = node_df['frame_id'].max()

    # x_min = np.round(data['x'].min() - 50)
    # x_max = np.round(data['x'].max() + 50)
    # y_min = np.round(data['y'].min() - 50)
    # y_max = np.round(data['y'].max() + 50)

    # data['x'] = data['x'] - x_min
    # data['y'] = data['y'] - y_min

    #TODO: create a way of obtaining the maximum timesteps 
    #TODO: input the appropriate dt 
    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name="test 1", aug_func=augment)

    # # Generate Maps
    # map_name = nusc.get('log', ns_scene['log_token'])['location']
    # nusc_map = NuScenesMap(dataroot=data_path, map_name=map_name)

    # type_map = dict()
    # x_size = x_max - x_min
    # y_size = y_max - y_min
    # patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
    # patch_angle = 0  # Default orientation where North is up
    # canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
    # homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    # layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
    #                'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
    # map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(
    #     np.uint8)
    # map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
    # # PEDESTRIANS
    # map_mask_pedestrian = np.stack((map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=0)
    # type_map['PEDESTRIAN'] = GeometricMap(data=map_mask_pedestrian, homography=homography, description=', '.join(layer_names))
    # # VEHICLES
    # map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)
    # type_map['VEHICLE'] = GeometricMap(data=map_mask_vehicle, homography=homography, description=', '.join(layer_names))

    # map_mask_plot = np.stack(((np.max(map_mask[:3], axis=0) - (map_mask[3] + 0.5 * map_mask[4]).clip(
    #     max=255)).clip(min=0).astype(np.uint8), map_mask[8], map_mask[9]), axis=0)
    # type_map['VISUALIZATION'] = GeometricMap(data=map_mask_plot, homography=homography, description=', '.join(layer_names))

    # scene.map = type_map
    # del map_mask
    # del map_mask_pedestrian
    # del map_mask_vehicle
    # del map_mask_plot

    #for node_id in pd.unique(data['node_id']):
        #for each particular object containing information of different frames & positions, obtain the x,y, velocity & acceleration
    node_frequency_multiplier = 1
    # node_df = data[data['node_id'] == node_id] #for node dataframe containing  the trajectory of objects in this dataframe

    if node_df['x'].shape[0] < 2:
        #continue
        print("Not feasible, only 1 position")
        #return

    if not np.all(np.diff(node_df['frame_id']) == 1):
        print('Occlusion')
        #continue  # TODO Make better
        #return 

    node_values = node_df[['x', 'y']].values
    x = node_values[:, 0]
    y = node_values[:, 1]
    #heading = node_df['heading'].values
    # if node_df.iloc[0]['type'] == env.NodeType.VEHICLE and not node_id == 'ego':
    #     # Kalman filter Agent
    #     vx = derivative_of(x, scene.dt)
    #     vy = derivative_of(y, scene.dt)
    #     velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

    #     filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
    #     P_matrix = None
    #     for i in range(len(x)):
    #         if i == 0:  # initalize KF
    #             # initial P_matrix
    #             P_matrix = np.identity(4)
    #         elif i < len(x):
    #             # assign new est values
    #             x[i] = x_vec_est_new[0][0]
    #             y[i] = x_vec_est_new[1][0]
    #             heading[i] = x_vec_est_new[2][0]
    #             velocity[i] = x_vec_est_new[3][0]

    #         if i < len(x) - 1:  # no action on last data
    #             # filtering
    #             x_vec_est = np.array([[x[i]],
    #                                     [y[i]],
    #                                     [heading[i]],
    #                                     [velocity[i]]])
    #             z_new = np.array([[x[i + 1]],
    #                                 [y[i + 1]],
    #                                 [heading[i + 1]],
    #                                 [velocity[i + 1]]])
    #             x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
    #                 x_vec_est=x_vec_est,
    #                 u_vec=np.array([[0.], [0.]]),
    #                 P_matrix=P_matrix,
    #                 z_new=z_new
    #             )
    #             P_matrix = P_matrix_new

    #     curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
    #     if pl < 1.0:  # vehicle is "not" moving
    #         x = x[0].repeat(max_timesteps + 1)
    #         y = y[0].repeat(max_timesteps + 1)
    #         heading = heading[0].repeat(max_timesteps + 1)
    #     global total
    #     global curv_0_2
    #     global curv_0_1
    #     total += 1
    #     if pl > 1.0:
    #         if curvature > .2:
    #             curv_0_2 += 1
    #             node_frequency_multiplier = 3*int(np.floor(total/curv_0_2))
    #         elif curvature > .1:
    #             curv_0_1 += 1
    #             node_frequency_multiplier = 3*int(np.floor(total/curv_0_1))

    vx = derivative_of(x, scene.dt)
    vy = derivative_of(y, scene.dt)
    ax = derivative_of(vx, scene.dt)
    ay = derivative_of(vy, scene.dt)

    # if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
    #     v = np.stack((vx, vy), axis=-1)
    #     v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
    #     heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
    #     heading_x = heading_v[:, 0]
    #     heading_y = heading_v[:, 1]

    #     data_dict = {('position', 'x'): x,
    #                     ('position', 'y'): y,
    #                     ('velocity', 'x'): vx,
    #                     ('velocity', 'y'): vy,
    #                     ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
    #                     ('acceleration', 'x'): ax,
    #                     ('acceleration', 'y'): ay,
    #                     ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
    #                     ('heading', 'x'): heading_x,
    #                     ('heading', 'y'): heading_y,
    #                     ('heading', '°'): heading,
    #                     ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
    #     node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
    # else:
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
    if node_df.iloc[0]['robot'] == True:
        node.is_robot = True
        scene.robot = node
    scene.nodes.append(node)

    return scene



#### Code for running the trajectron for mpc model
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


def main():
    # Choose one of the model directory names under the experiment/*/models folders.
    # Possibilities are 'vel_ee', 'int_ee', 'int_ee_me', or 'robot'
    model_dir = os.path.join(args.log_dir, 'int_ee_cassie')

    # Load hyperparameters from json
    config_file = os.path.join(model_dir, args.conf)
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

    output_save_dir = os.path.join(model_dir, 'pred_figs')
    pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)

    ##this is the main part > to open the dill files and run them in, obtain all the environment & scenes from the particular file
    # eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
    # with open(eval_data_path, 'rb') as f:
    #     eval_env = dill.load(f, encoding='latin1')
    frame_id = [i for i in range(0,21)]
    x_pos_list = [i for i in range(0,21)]
    y_pos_list = [i for i in range(0,21)]
    #print(frame_id)
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE
    

    node_df = pd.DataFrame({'frame_id': frame_id,
                                        'type': env.NodeType.PEDESTRIAN ,
                                        #'node_id': annotation['instance_token'],
                                        'robot': False,
                                        'x': x_pos_list,
                                        'y': y_pos_list,
                                        'z': 0,
                                        'length': 0.4,
                                        'width': 0.5,
                                        'height': 0.3,
                                        'heading': 0})

    done_scene=generate_env_with_scene_from_dataframe(node_df,0.5,env)
    scenes = []
    scenes.append(done_scene)
    env.scenes=scenes
    eval_env=env

    #obtain the environment containing multiple scenes from a txt file
    
    #add a particular robot to each scene in data struct scenes
    if eval_env.robot_type is None and hyperparams['incl_robot_node']:
        eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in eval_env.scenes:
            scene.add_robot_from_nodes(eval_env.robot_type)

    # print('Loaded data from %s' % (eval_data_path,))

    # Creating a dummy environment with a single scene that contains information about the world.
    # When using this code, feel free to use whichever scene index or initial timestep you wish.
    scene_idx = 0

    # You need to have at least acceleration, so you want 2 timesteps of prior data, e.g. [0, 1],
    # so that you can immediately start incremental inference from the 3rd timestep onwards.
    init_timestep = 1

    eval_scene = eval_env.scenes[scene_idx]

    print(eval_scene)

    # creating a singular environemnt from a particular scene within 
    # collection of scenes within the env datasets
    online_env = create_online_env(eval_env, hyperparams, scene_idx, init_timestep)

    #combine & load model, possible to skip this step
    model_registrar = ModelRegistrar(model_dir, args.eval_device)
    model_registrar.load_models(iter_num=12)

    trajectron = OnlineTrajectron(model_registrar,
                                  hyperparams,
                                  args.eval_device)

    # If you want to see what different robot futures do to the predictions, uncomment this line as well as
    # related "... += adjustment" lines below.
    # adjustment = np.stack([np.arange(13)/float(i*2.0) for i in range(6, 12)], axis=1)

    # Here's how you'd incrementally run the model, e.g. with streaming data; setting the environment
    # accordingly for the trajectron to make predictions 
    trajectron.set_environment(online_env, init_timestep)

    #this part is important for the visualisation of future trajectories
    # We are working within only 1 particular scene and we are essentially calling eval_scene

    for timestep in range(init_timestep + 1, eval_scene.timesteps):
        #TODO: for each timestep, initialise a dictionary specific to the timestep
        prediction_dict_at_timestep=dict()

        input_dict = eval_scene.get_clipped_input_dict(timestep, hyperparams['state']) #get the different pedestrian positions & details of the scene 
    
        maps = None

        if hyperparams['use_map_encoding']:
            #configure to use maps
            maps = get_maps_for_input(input_dict, eval_scene, hyperparams)

        robot_present_and_future = None

        #for each time step, add the trajectory of the robot in the form of an array
        # Keane: You might be able to input the trajectory of your own robot
        
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

        ##@TODO: Try the trajectron prediction of entire distribution
        prediction_horizon = 6
        dists, preds = trajectron.incremental_forward(input_dict,
                                                      maps,
                                                      prediction_horizon=prediction_horizon, #how long into the future do you want to predict?
                                                      num_samples=4, #how many different trajectories do you want?
                                                      robot_present_and_future=robot_present_and_future,
                                                      full_dist=True)
        end = time.time()
        print("t=%d: took %.2f s (= %.2f Hz) w/ %d nodes and %d edges" % (timestep, end - start,
                                                                          1. / (end - start), len(trajectron.nodes),
                                                                          trajectron.scene_graph.get_num_edges()))
        for node in eval_scene.nodes:
            if node in preds:
                raw_predictions = preds[node] 
                
        for p_horz in range(raw_predictions.shape[2]): 
            new_key = "t" + str(timestep+1+p_horz)
            all_x_at_timestep = []
            all_y_at_timestep = []
            for traj_n in range(raw_predictions.shape[1]):
                all_x_at_timestep.append(raw_predictions[0,traj_n,p_horz,0]) #obtains x value at each timestep
                all_y_at_timestep.append(raw_predictions[0,traj_n,p_horz,1]) #obtains y value at each timestep
            prediction_dict_at_timestep[new_key] = [all_x_at_timestep,all_y_at_timestep]

        print(prediction_dict_at_timestep) 
        """Dictionary is in the form of 
        {timestep i:[[each x corresponding to diff trajectories],[each y corresponding to diff trajectories]]}
        """

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

        fig.savefig(os.path.join(output_save_dir, f'pred_{timestep}.pdf'), dpi=300)
        plt.close(fig)
        break


if __name__ == '__main__':
    main()


