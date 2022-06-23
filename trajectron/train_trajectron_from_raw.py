import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import pandas as pd
from environment import Environment, Scene, Node, derivative_of
import warnings
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import EnvironmentDataset, collate
from tensorboardX import SummaryWriter
# torch.autograd.set_detect_anomaly(True)

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
    args.eval_device = torch.device('cpu')

# This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
#torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

### TODO: Custom functions to train risk trajectron on txt file directly
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
    },
    'ROBOT': {
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



def obtain_scene_from_txt(txt_filename):
    ## TODO: Add robot to node_type_list
    # env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    env = Environment(node_type_list=['PEDESTRIAN','ROBOT'], standardization=standardization)
    attention_radius = dict()

    ## TODO: Add the so called criterion; change robot-pedestrian to CVAR
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    attention_radius[(env.NodeType.ROBOT,env.NodeType.PEDESTRIAN)] = 0    
    attention_radius[(env.NodeType.PEDESTRIAN,env.NodeType.ROBOT)] = 0    
    attention_radius[(env.NodeType.ROBOT,env.NodeType.ROBOT)] = 1.0
    env.attention_radius = attention_radius

    ### Configure the robot type
    env.robot_type = env.NodeType.ROBOT

    scenes = []
    
    print('At',txt_filename)

    data = pd.read_csv(txt_filename, sep='\t', index_col=False, header=None)
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
    initialise_robot = True

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

        data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
        node_data = pd.DataFrame(data_dict, columns=data_columns)
        #TODO: Replace random node_id with NodeType Robot
        if initialise_robot:
            print(node_id)
            node = Node(node_type=env.NodeType.ROBOT, node_id=node_id, data=node_data)
            node.is_robot = True
            scene.robot = node
            initialise_robot = False
        
        else:
            node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
        node.first_timestep = new_first_idx

        scene.nodes.append(node)
    if 'train' in txt_filename:
        scene.augmented = list()
        angles = np.arange(0, 360, 15) if 'train' in txt_filename else [0]
        for angle in angles:
            scene.augmented.append(augment_scene(scene, angle))

    print(scene)

    #after generating the scene, appending all the scenes to the data struct
    #environment can contain scenes which contains many scenes of diff traj
    scenes.append(scene)
    env.scenes = scenes
    return env



#################################
##### New functions above #######
#################################

def main():
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
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
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    # print('| robot node: %s' % args.incl_robot_node)
    print('| robot node: Yes')
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    
    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)


    ### TODO: change this to removing the dill and training directly from one eth txt file

    # Load training and evaluation environments and scenes
    train_scenes = []
    #train_data_path = os.path.join(args.data_dir, args.train_data_dict)

   
    # with open(train_data_path, 'rb') as f:
    #     train_env = dill.load(f, encoding='latin1')

    train_txt_file = "/home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/train/crowds_zara01_train.txt"
    eval_txt_file = "/home/kong35/Trajectron-plus-plus/experiments/pedestrians/raw/eth/val/crowds_zara01_val.txt"
    train_env = obtain_scene_from_txt(train_txt_file)

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


    ### Only for adding a particular robot if there is no robot node 
    ### i.e. for pedestrian only datasets
    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=True) #trained on pedestrian-only
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=0)#args.preprocess_workers)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_txt_file}")

    ## Scenes for evalation (may not be so important as of now)
    eval_scenes = []
    eval_scenes_sample_probs = None

    if args.eval_every is not None:
        # eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        # with open(eval_data_path, 'rb') as f:
        #     eval_env = dill.load(f, encoding='latin1')
        eval_env = obtain_scene_from_txt(eval_txt_file)

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if args.eval_device is 'cpu' else True,
                                                         batch_size=args.eval_batch_size,
                                                         shuffle=True,
                                                         num_workers=0) #args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_txt_file}")

    ### TODO: calculate an offline scene graph but change the attention radius into something else
    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")

    model_registrar = ModelRegistrar(model_dir, args.device)

    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)

    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()
    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')

    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])

    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    for epoch in range(1, args.train_epochs + 1):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        for node_type, data_loader in train_data_loader.items():
            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80)
            for batch in pbar:
                trajectron.set_curr_iter(curr_iter)
                trajectron.step_annealers(node_type)
                optimizer[node_type].zero_grad()
                #print(batch)
                train_loss = trajectron.train_loss(batch, node_type)
                pbar.set_description(f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f}")
                train_loss.backward()
                # Clipping gradients.
                if hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                lr_scheduler[node_type].step()

                if not args.debug:
                    log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                          lr_scheduler[node_type].get_lr()[0],
                                          curr_iter)
                    log_writer.add_scalar(f"{node_type}/train/loss", train_loss, curr_iter)

                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter
        train_dataset.augment = False
        if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################

        ### To visualise, enter: tensorboard --logdir=/home/kong35/Trajectron-plus-plus/experiments/pedestrians/models/models_20_Jun_2022_13_16_30_eth_vel_ar3
        
        if args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict random timestep to plot for train data set ### Essentially, we are picking random timesteps at which we plot
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = trajectron.predict(scene,
                                                 timestep,
                                                 ph,
                                                 min_future_timesteps=ph,
                                                 z_mode=True,
                                                 gmm_mode=True,
                                                 all_z_sep=False,
                                                 full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/prediction', fig, epoch)

                model_registrar.to(args.eval_device)
                # Predict random timestep to plot for eval data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(eval_scenes, p=eval_scenes_sample_probs)
                else:
                    scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      num_samples=20,
                                                      min_future_timesteps=ph,
                                                      z_mode=False,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction', fig, epoch)

                # Predict random timestep to plot for eval data set
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      min_future_timesteps=ph,
                                                      z_mode=True,
                                                      gmm_mode=True,
                                                      all_z_sep=True,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    eval_loss = []
                    print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(batch, node_type)
                        pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                        eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
                        del batch

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=50,
                                                          min_future_timesteps=ph,
                                                          full_dist=False)

                    eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                 scene.dt,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 node_type_enum=eval_env.NodeType,
                                                                                 map=scene.map))

                evaluation.log_batch_errors(eval_batch_errors,
                                            log_writer,
                                            'eval',
                                            epoch,
                                            bar_plot=['kde'],
                                            box_plot=['ade', 'fde'])

                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                eval_batch_errors_ml = []
                for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(scene.timesteps)

                    predictions = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=1,
                                                          min_future_timesteps=ph,
                                                          z_mode=True,
                                                          gmm_mode=True,
                                                          full_dist=False)

                    eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                                                                                    scene.dt,
                                                                                    max_hl=max_hl,
                                                                                    ph=ph,
                                                                                    map=scene.map,
                                                                                    node_type_enum=eval_env.NodeType,
                                                                                    kde=False))

                evaluation.log_batch_errors(eval_batch_errors_ml,
                                            log_writer,
                                            'eval/ml',
                                            epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch)


if __name__ == '__main__':
    main()
