#from math import dist
from sqlite3 import Time
from statistics import stdev
from turtle import distance, position
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.signal as ss
from collections import defaultdict
import warnings
from .node import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import scipy.stats
from scipy.stats import norm
from scipy.spatial import distance


 ### New Functions ###

def GP(obs_array):
    """Goal is to take in an array of obstacles and obtain the mean & std of the next 
    timestep
    """
    
    #print(len(obs_array))
    prev_timesteps = np.arange(0, len(obs_array), 1).reshape(-1,1)
    #print(prev_timesteps)
    future_timesteps = np.arange(len(obs_array),len(obs_array)+1,1).reshape(-1,1)
    #print(future_timesteps)

    kernel =  1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    #skip the NaN values
    gaussian_process.fit(prev_timesteps, obs_array)
    gaussian_process.kernel_
    mean_val, std_val = gaussian_process.predict(future_timesteps, return_std=True)
    #print(mean_val)
    #print(std_val)
    return mean_val[0], std_val[0]

def perform_GP(scene_temp_dict,node_id):
    # For now, the robot will always take a node_id of 1
    node_idx = node_id -1
    pos_values = scene_temp_dict[list(scene_temp_dict.keys())[node_idx]]  
    
    all_x_vals = pos_values[:,0].reshape(-1,1)
    all_y_vals = pos_values[:,1].reshape(-1,1)

    total_val_count = np.sum(~np.isnan(all_x_vals))
    all_gauss_x = []
    all_gauss_y = []
    for i in range(1,total_val_count+1):
        gauss_x = GP(all_x_vals[:i,0])
        all_gauss_x.append(gauss_x)
        gauss_y = GP(all_y_vals[:i,0])
        all_gauss_y.append(gauss_y)
    return {'x_gauss' : all_gauss_x,'y_gauss' : all_gauss_y}

def perform_GP2(scene_temp_dict,node_id,curr_timestep):
    # For now, the robot will always take a node_id of 1
    node_idx = node_id -1
    pos_values = scene_temp_dict[list(scene_temp_dict.keys())[node_idx]]  
    
    all_x_vals = pos_values[:curr_timestep+1,0].reshape(-1,1)
    all_y_vals = pos_values[:curr_timestep+1,1].reshape(-1,1)
    print(all_x_vals)

    return GP(all_x_vals),GP(all_y_vals)

def Var(beta, dist, std):
    q = 1-beta
    var = norm.ppf(q,dist,std)
	#tic = time.perf_counter()
    cvar = (1/(1-q))*scipy.stats.norm.expect(lambda x: x, lb = var)
	#toc = time.perf_counter()
	#print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
    return cvar

### New functions above ###


class Edge(object):
    def __init__(self, curr_node, other_node):
        self.id = self.get_edge_id(curr_node, other_node)
        self.type = self.get_edge_type(curr_node, other_node)
        self.curr_node = curr_node
        self.other_node = other_node

    @staticmethod
    def get_edge_id(n1, n2):
        raise NotImplementedError("Use one of the Edge subclasses!")

    @staticmethod
    def get_str_from_types(nt1, nt2):
        raise NotImplementedError("Use one of the Edge subclasses!")

    @staticmethod
    def get_edge_type(n1, n2):
        raise NotImplementedError("Use one of the Edge subclasses!")

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.id == other.id)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return self.id


class UndirectedEdge(Edge):
    def __init__(self, curr_node, other_node):
        super(UndirectedEdge, self).__init__(curr_node, other_node)

    @staticmethod
    def get_edge_id(n1, n2):
        return '-'.join(sorted([str(n1), str(n2)]))

    @staticmethod
    def get_str_from_types(nt1, nt2):
        return '-'.join(sorted([nt1.name, nt2.name]))

    @staticmethod
    def get_edge_type(n1, n2):
        return '-'.join(sorted([n1.type.name, n2.type.name]))


class DirectedEdge(Edge):
    def __init__(self, curr_node, other_node):
        super(DirectedEdge, self).__init__(curr_node, other_node)

    @staticmethod
    def get_edge_id(n1, n2):
        return '->'.join([str(n1), str(n2)])

    @staticmethod
    def get_str_from_types(nt1, nt2):
        return '->'.join([nt1.name, nt2.name])

    @staticmethod
    def get_edge_type(n1, n2):
        return '->'.join([n1.type.name, n2.type.name])


class TemporalSceneGraph(object):
    def __init__(self,
                 edge_radius,
                 nodes=None,
                 adj_cube=np.zeros((1, 0, 0)),
                 weight_cube=np.zeros((1, 0, 0)),
                 node_type_mat=np.zeros((0, 0)),
                 edge_scaling=None):
        self.edge_radius = edge_radius
        self.nodes = nodes
        if nodes is None:
            self.nodes = np.array([])
        self.adj_cube = adj_cube
        self.weight_cube = weight_cube
        self.node_type_mat = node_type_mat
        self.adj_mat = np.max(self.adj_cube, axis=0).clip(max=1.0)
        self.edge_scaling = edge_scaling
        self.node_index_lookup = None
        self.calculate_node_index_lookup()

    def calculate_node_index_lookup(self):
        node_index_lookup = dict()
        for i, node in enumerate(self.nodes):
            node_index_lookup[node] = i

        self.node_index_lookup = node_index_lookup

    def get_num_edges(self, t=0):
        return np.sum(self.adj_cube[t]) // 2

    def get_index(self, node):
        return self.node_index_lookup[node]
    
   
    @classmethod
    def create_from_temp_scene_dict(cls,
                                    scene_temp_dict,
                                    attention_radius,
                                    duration=1,
                                    edge_addition_filter=None,
                                    edge_removal_filter=None,
                                    online=False):
        """
        Construct a spatiotemporal graph from node positions in a dataset.

        :param scene_temp_dict: Dict with all nodes in scene as keys and np.ndarray with positions as value
        :param attention_radius: Attention radius dict.
        :param duration: Temporal duration of the graph.
        :param edge_addition_filter: -
        :param edge_removal_filter: -
        :return: TemporalSceneGraph
        """

        nodes = scene_temp_dict.keys()
        N = len(nodes)
        total_timesteps = duration

        if N == 0:
            return TemporalSceneGraph(attention_radius)

        position_cube = np.full((total_timesteps, N, 2), np.nan)
        ## position_cube has dimensions of (total timesteps ,number of nodes, 2), and it is filled with all NaN values

        #print(position_cube.shape)

        adj_cube = np.zeros((total_timesteps, N, N), dtype=np.int8)
        dist_cube = np.zeros((total_timesteps, N, N), dtype=np.float)

        node_type_mat = np.zeros((N, N), dtype=np.int8)
        node_attention_mat = np.zeros((N, N), dtype=np.float)

        ### TODO: 1 => scene_temp_dict contains different nodes and their x and y positions for their different timesteps (710 in total)
        ### GP function should take in the dict & the node_id or node_type and produce future prediction/ uncertainty based on past 
        ## obtain values: current & past for each timestep
        ### feed into GP 
        ### obtain mean and variance
        #print(scene_temp_dict)
        robot_node_id = 1
        robot_gauss_dict = perform_GP(scene_temp_dict,robot_node_id) 

        for node_idx, node in enumerate(nodes):
            if online:
                # RingBuffers do not have a fixed constant size. Instead, they grow up to their capacity. Thus,
                # we need to fill the values preceding the RingBuffer values with NaNs to make them fill the
                # position_cube.
                position_cube[-scene_temp_dict[node].shape[0]:, node_idx] = scene_temp_dict[node]
            else:
                ## Initialising the position of each node as a numpy array, represented as columns x y node index
                ## Structure here for position_cube is (timestep, nodes, 2), where the 2 corresponds to the position
                ## Essentially, what you are doing is: For all timesteps groups, select node index(iterate down each row of timestep group),
                ## and assign to x,y position
                
                position_cube[:, node_idx] = scene_temp_dict[node]
                
            
            node_type_mat[:, node_idx] = node.type.value

            for node_idx_from, node_from in enumerate(nodes):
                ## Initialising numpy array containing information of the node types coresponding to different node indexes

                ### Pattern for the creation of node_attention_mat
                # row position: from particular node; column position: to particular node
                # i.e. r1c2: attention radius from from 1 to 2
                node_attention_mat[node_idx_from, node_idx] = attention_radius[(node_from.type, node.type)]
        np.fill_diagonal(node_type_mat, 0)

        for timestep in range(position_cube.shape[0]):

            #### TODO:Segment to change; create new function based on cVAR ####


            ### Calculation of the distance between each nodes and then you put it as a new array 
            print(position_cube[timestep])
            # timestep is zero
            ### Position cube[timestep] gives us each row of x,y position corresponding to each node
            # row 1 belongs to node 1 and so on and so forth 
            dists = squareform(pdist(position_cube[timestep], metric='euclidean'))
            #print(dists.shape)
            print(dists)
            ### TODO: 2 => modify dists to CVaR values; (set condition only if distance is below a particular value?)
            ### dists is in the form of row pos: from node; col pos: to node
            ### hence, what you would do is to change those positions involving 
            ### the node id of robot to cVAR 
            ## for now: node id 1 is robot therefore all entries of row 1 & column 1 corresponds to robot involvement
            robo_col= robot_node_id-1
            risk_value = 0.02
            beta = 0.05
            after_CVaR_lst = []
            for other_node_row in range(N):
                dist = dists[other_node_row,robo_col]
                critical_dist = 1
                if dist > critical_dist:
                    if timestep != position_cube.shape[0]:
                        new_node_positions = position_cube[timestep+1]
                        print(new_node_positions)
                        #print(new_node_positions.shape)
                        other_node_pos = new_node_positions[other_node_row]
                        x, y = perform_GP2(scene_temp_dict,robot_node_id,timestep)
                        GP_pos = [x[0],y[0]]
                        # now calculate the CVaR between other_node_pos and Gaussian Prediction
                        var_dist= distance.euclidean(GP_pos,other_node_pos)
                        GP_std = x[1]+y[1]
                        CVaR = CVaR(beta,var_dist,GP_std)
                        after_CVaR_lst.append([robo_col,other_node_row,1 if CVaR > risk_value else 0])
            
            # Put a 1 for all agent pairs which are closer than the edge_radius.
            # Can produce a warning as dists can be nan if no data for node is available.
            # This is accepted as nan <= x evaluates to False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ### node_attention_mat is the criteria in which we generate our edges from 
                ### TODO: if lower than attention criteria, then we compute using cVAR
                ### TODO: change node_attention_mat to cVAR 
                ### For each number in the dists matrix, if it is lower than the same position in the node_attention_mat,
                ### return 1, otherwise return 0
                adj_matrix = (dists <= node_attention_mat).astype(np.int8) * node_type_mat
            
            for robo_col, other_node_row, info in after_CVaR_lst:
                adj_matrix[robo_col][other_node_row] = info
            print(adj_matrix)
            break
            # Remove self-loops.; obtaining adjacency matrix and using this for edge generation
            np.fill_diagonal(adj_matrix, 0)
            
            adj_cube[timestep] = adj_matrix
            dist_cube[timestep] = dists

        dist_cube[np.isnan(dist_cube)] = 0.
        weight_cube = np.divide(1.,
                                dist_cube,
                                out=np.zeros_like(dist_cube),
                                where=(dist_cube > 0.))
        edge_scaling = None
        if edge_addition_filter is not None and edge_removal_filter is not None:
            edge_scaling = cls.calculate_edge_scaling(adj_cube, edge_addition_filter, edge_removal_filter)
        tsg = cls(attention_radius,
                  np.array(list(nodes)),
                  adj_cube, weight_cube,
                  node_type_mat,
                  edge_scaling=edge_scaling)
        return tsg

    @staticmethod
    def calculate_edge_scaling(adj_cube, edge_addition_filter, edge_removal_filter):
        shifted_right = np.pad(adj_cube, ((len(edge_addition_filter) - 1, 0), (0, 0), (0, 0)), 'constant', constant_values=0)

        new_edges = np.minimum(
            ss.convolve(shifted_right, np.reshape(edge_addition_filter, (-1, 1, 1)), 'full'), 1.
        )[(len(edge_addition_filter) - 1):-(len(edge_addition_filter) - 1)]

        new_edges[adj_cube == 0] = 0

        result = np.minimum(
            ss.convolve(new_edges, np.reshape(edge_removal_filter, (-1, 1, 1)), 'full'), 1.
        )[:-(len(edge_removal_filter) - 1)]

        return result

    def to_scene_graph(self, t, t_hist=0, t_fut=0):
        """
        Creates a Scene Graph from a Temporal Scene Graph

        :param t: Time in Temporal Scene Graph for which Scene Graph is created.
        :param t_hist: Number of history timesteps which are considered to form edges in Scene Graph.
        :param t_fut: Number of future timesteps which are considered to form edges in Scene Graph.
        :return: SceneGraph
        """
        lower_t = np.clip(t-t_hist, a_min=0, a_max=None)
        higher_t = np.clip(t + t_fut + 1, a_min=None, a_max=self.adj_cube.shape[0] + 1)
        adj_mat = np.max(self.adj_cube[lower_t:higher_t], axis=0)
        weight_mat = np.max(self.weight_cube[lower_t:higher_t], axis=0)
        return SceneGraph(self.edge_radius,
                          self.nodes,
                          adj_mat,
                          weight_mat,
                          self.node_type_mat,
                          self.node_index_lookup,
                          edge_scaling=self.edge_scaling[t] if self.edge_scaling is not None else None)


class SceneGraph(object):
    def __init__(self,
                 edge_radius,
                 nodes=None,
                 adj_mat=np.zeros((0, 0)),
                 weight_mat=np.zeros((0, 0)),
                 node_type_mat=np.zeros((0, 0)),
                 node_index_lookup=None,
                 edge_scaling=None):
        self.edge_radius = edge_radius
        self.nodes = nodes
        if nodes is None:
            self.nodes = np.array([])
        self.node_type_mat = node_type_mat
        self.adj_mat = adj_mat
        self.weight_mat = weight_mat
        self.edge_scaling = edge_scaling
        self.node_index_lookup = node_index_lookup

    def get_index(self, node):
        return self.node_index_lookup[node]

    def get_num_edges(self):
        return np.sum(self.adj_mat) // 2

    def get_neighbors(self, node, node_type):
        """
        Get all neighbors of a node.

        :param node: Node for which all neighbors are returned.
        :param node_type: Specifies node types which are returned.
        :return: List of all neighbors.
        """
        node_index = self.get_index(node)
        connection_mask = self.get_connection_mask(node_index)
        mask = ((self.node_type_mat[node_index] == node_type.value) * connection_mask)
        return self.nodes[mask]

    def get_edge_scaling(self, node=None):
        if node is None:
            return self.edge_scaling
        else:
            node_index = self.get_index(node)
            connection_mask = self.get_connection_mask(node_index)
            return self.edge_scaling[node_index, connection_mask]

    def get_edge_weight(self, node=None):
        if node is None:
            return self.weight_mat
        else:
            node_index = self.get_index(node)
            connection_mask = self.get_connection_mask(node_index)
            return self.weight_mat[node_index, connection_mask]

    def get_connection_mask(self, node_index):
        if self.edge_scaling is None: # We do not use edge scaling
            return self.adj_mat[node_index] > 0.
        else:
            return self.edge_scaling[node_index] > 1e-2

    def __sub__(self, other):
        new_nodes = [node for node in self.nodes if node not in other.nodes]
        removed_nodes = [node for node in other.nodes if node not in self.nodes]

        our_types = set(node.type for node in self.nodes)
        other_types = set(node.type for node in other.nodes)
        all_node_types = our_types | other_types

        new_neighbors = defaultdict(lambda: defaultdict(set))
        for node in self.nodes:
            if node in removed_nodes:
                continue

            if node in other.nodes:
                for node_type in all_node_types:
                    new_items = set(self.get_neighbors(node, node_type)) - set(other.get_neighbors(node, node_type))
                    if len(new_items) > 0:
                        new_neighbors[node][DirectedEdge.get_edge_type(node, Node(node_type, None, None))] = new_items
            else:
                for node_type in our_types:
                    neighbors = self.get_neighbors(node, node_type)
                    if len(neighbors) > 0:
                        new_neighbors[node][DirectedEdge.get_edge_type(node, Node(node_type, None, None))] = set(neighbors)

        removed_neighbors = defaultdict(lambda: defaultdict(set))
        for node in other.nodes:
            if node in removed_nodes:
                continue

            if node in self.nodes:
                for node_type in all_node_types:
                    removed_items = set(other.get_neighbors(node, node_type)) - set(self.get_neighbors(node, node_type))
                    if len(removed_items) > 0:
                        removed_neighbors[node][DirectedEdge.get_edge_type(node, Node(node_type, None, None))] = removed_items
            else:
                for node_type in other_types:
                    neighbors = other.get_neighbors(node, node_type)
                    if len(neighbors) > 0:
                        removed_neighbors[node][DirectedEdge.get_edge_type(node, Node(node_type, None, None))] = set(neighbors)

        return new_nodes, removed_nodes, new_neighbors, removed_neighbors


if __name__ == '__main__':
    from environment import NodeTypeEnum
    import time

    # # # # # # # # # # # # # # # # #
    # Testing edge mask calculation #
    # # # # # # # # # # # # # # # # #
    B = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                  [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]])[:, :, np.newaxis, np.newaxis]
    print(B.shape)

    edge_addition_filter = [0.25, 0.5, 0.75, 1.0]
    edge_removal_filter = [1.0, 0.5, 0.0]
    for i in range(B.shape[0]):
        A = B[i]  # (time, N, N)

        print(A[:, 0, 0])

        start = time.time()
        new_edges = np.minimum(ss.convolve(A, np.reshape(edge_addition_filter, (-1, 1, 1)), 'full'), 1.)[(len(edge_addition_filter) - 1):]
        old_edges = np.minimum(ss.convolve(A, np.reshape(edge_removal_filter, (-1, 1, 1)), 'full'), 1.)[:-(len(edge_removal_filter) - 1)]
        res = np.minimum(new_edges + old_edges, 1.)[:, 0, 0]
        end = time.time()
        print(end - start)
        print(res)

        start = time.time()
        res = TemporalSceneGraph.calculate_edge_scaling(A, edge_addition_filter, edge_removal_filter)[:, 0, 0]
        end = time.time()
        print(end - start)
        print(res)

        print('-'*40)

    # # # # # # # # # # # # # # #
    # Testing graph subtraction #
    # # # # # # # # # # # # # # #
    print('\n' + '-' * 40 + '\n')

    node_type_list = ['PEDESTRIAN',
                      'BICYCLE',
                      'VEHICLE']
    nte = NodeTypeEnum(node_type_list)

    attention_radius = dict()
    attention_radius[(nte.PEDESTRIAN, nte.PEDESTRIAN)] = 5.0
    attention_radius[(nte.PEDESTRIAN, nte.VEHICLE)] = 20.0
    attention_radius[(nte.PEDESTRIAN, nte.BICYCLE)] = 10.0
    attention_radius[(nte.VEHICLE, nte.PEDESTRIAN)] = 20.0
    attention_radius[(nte.VEHICLE, nte.VEHICLE)] = 20.0
    attention_radius[(nte.VEHICLE, nte.BICYCLE)] = 20.0
    attention_radius[(nte.BICYCLE, nte.PEDESTRIAN)] = 10.0
    attention_radius[(nte.BICYCLE, nte.VEHICLE)] = 20.0
    attention_radius[(nte.BICYCLE, nte.BICYCLE)] = 10.0

    scene_dict1 = {Node(nte.PEDESTRIAN, node_id='1'): np.array([1, 0]),
                   Node(nte.PEDESTRIAN, node_id='2'): np.array([0, 1])}
    sg1 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict1,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    scene_dict2 = {Node(nte.PEDESTRIAN, node_id='1'): np.array([1, 0]),
                   Node(nte.PEDESTRIAN, node_id='2'): np.array([1, 1])}
    sg2 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict2,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)

    # # # # # # # # # # # # # # #
    print('\n' + '-' * 40 + '\n')

    scene_dict1 = {Node(nte.PEDESTRIAN, node_id='1'): np.array([1, 0]),
                   Node(nte.PEDESTRIAN, node_id='2'): np.array([0, 1])}
    sg1 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict1,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    scene_dict2 = {Node(nte.PEDESTRIAN, node_id='1'): np.array([1, 0]),
                   Node(nte.PEDESTRIAN, node_id='2'): np.array([1, 1]),
                   Node(nte.PEDESTRIAN, node_id='3'): np.array([20, 1])}
    sg2 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict2,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)

    # # # # # # # # # # # # # # #
    print('\n' + '-' * 40 + '\n')

    scene_dict1 = {Node(nte.PEDESTRIAN, node_id='1'): np.array([1, 0]),
                   Node(nte.PEDESTRIAN, node_id='2'): np.array([0, 1])}
    sg1 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict1,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    scene_dict2 = {Node(nte.PEDESTRIAN, node_id='1'): np.array([1, 0]),
                   Node(nte.PEDESTRIAN, node_id='2'): np.array([10, 1]),
                   Node(nte.PEDESTRIAN, node_id='3'): np.array([20, 1])}
    sg2 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict2,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)

    # # # # # # # # # # # # # # #
    print('\n' + '-' * 40 + '\n')

    scene_dict1 = {Node(nte.PEDESTRIAN, node_id='1'): np.array([0, 0]),
                   Node(nte.PEDESTRIAN, node_id='2'): np.array([0, 1])}
    sg1 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict1,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    scene_dict2 = {Node(nte.PEDESTRIAN, node_id='2'): np.array([10, 1]),
                   Node(nte.PEDESTRIAN, node_id='3'): np.array([12, 1]),
                   Node(nte.PEDESTRIAN, node_id='4'): np.array([13, 1])}
    sg2 = TemporalSceneGraph.create_from_temp_scene_dict(
        scene_dict2,
        attention_radius=attention_radius,
        duration=1,
        edge_addition_filter=[0.25, 0.5, 0.75, 1.0],
        edge_removal_filter=[1.0, 0.0]).to_scene_graph(t=0)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)
