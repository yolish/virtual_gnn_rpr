from torch_geometric.nn.conv import GATConv
import torch.nn as nn
import torch

def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)

    return q


def qinv(q):
    """
    Inverts a unit quaternion
    :param q: (torch.tensor) Nx4 tensor (unit quaternion)
    :return: Nx4 tensor (inverse quaternion)
    """
    q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
    return q_inv


def compute_rel_pose(abs_pose_query, abs_pose_neighbor):
    rel_x = abs_pose_query[:, :3] - abs_pose_neighbor[:, :3]
    rel_q = qmult(abs_pose_query[:, 3:], abs_pose_neighbor[:, 3:])
    return torch.cat((rel_x, rel_q), dim=1)


def compute_abs_pose(rel_x, rel_q, edge_atts, neighbor_poses):
    num_query, num_neighbors = edge_atts.shape
    abs_pose = torch.zeros((num_query, 7)).to(neighbor_poses.dtype).to(neighbor_poses.device)
    for i in range(num_query):
        indices = list(range(i*num_neighbors,(i+1)*num_neighbors))
        query_neighbor_poses = neighbor_poses[indices, :]
        query_rel_x = rel_x[indices, :]
        query_rel_q = rel_q[indices, :]
        pose_candidates = torch.zeros_like(query_neighbor_poses)
        for j in range(num_neighbors):
            nbr_pose = query_neighbor_poses[j]
            pose_candidates[j, :3] = nbr_pose[:3] + query_rel_x[j]
            pose_candidates[j, 3:] = qmult(query_rel_q[j].unsqueeze(0), qinv(nbr_pose[3:].unsqueeze(0))).squeeze(0)
        abs_pose[i, :] = torch.sum(pose_candidates*edge_atts[i].unsqueeze(1), dim=0)
        abs_pose[i, 3:] = abs_pose[i, 3:]/torch.norm(abs_pose[i, 3:])
    return abs_pose


class RPGNN(nn.Module):
    def __init__(self, config):

        super(RPGNN, self).__init__()

        gnn_config = config.get("gnn_rpr")
        num_layers = gnn_config.get("num_layers")
        in_channels = gnn_config.get("in_dim")
        out_channels = gnn_config.get("out_dim")
        num_heads = gnn_config.get("num_heads")
        dropout = gnn_config.get("dropout")

        self.gat = nn.ModuleList()
        for _ in range(num_layers-1):
            self.gat.append(GATConv(in_channels=in_channels,
                                    out_channels=in_channels,
                                    concat=False,
                                    heads=num_heads,
                                    dropout=dropout))

        self.gat.append(GATConv(in_channels=in_channels,
                                    out_channels=out_channels,
                                concat=False,
                                    heads=num_heads,
                                    dropout=dropout))

        self.edge_feature_updator = nn.Sequential(nn.Linear(out_channels * 2, out_channels),
                                                  nn.ReLU(),
                                                  nn.Linear(out_channels, out_channels),
                                                  nn.ReLU())

        self.x_regressor = nn.Linear(out_channels, 3)
        self.q_regressor = nn.Linear(out_channels, 4)

        #todo reset params

    def forward(self, query_encoding, neighbor_encoding, neighbor_poses=None):
        '''
        query_encoding torch.Tensor [N X in_channels]
        neighbor_encoding torch.Tensor [N*k x in_channels]
        neighbor_poses torch.Tensor [N x k x 7] optional
        '''
        num_query_nodes = query_encoding.shape[0]  # N
        num_neighbors = neighbor_encoding.shape[0] // num_query_nodes  # k (closest for each query)

        # Node features
        node_features = torch.cat((query_encoding, neighbor_encoding), dim=0)
        # 0..N-1 are query features
        # N ... N+k-1 are the features of the neighbors of the first query
        # N+k ... N+2k-1 are the features of the neighbors of the second query
        # ...
        # N + (k*i) ... N*(k*(i+1)) -1 are the featurs of the neighbors of the i+1 query

        # Build the graph in torch geometric format
        # Compute edge index
        edge_index = []
        for i in range(num_query_nodes):

            neighbor_indices = list(range(num_query_nodes+(num_neighbors*i),num_query_nodes+(num_neighbors*(i+1))))
            # Add the edge indices and compute the edge features
            # from query to neighbors (in both directions)
            for j in range(num_neighbors):
                nbr_idx = neighbor_indices[j]
                edge_index.append([i, nbr_idx])
                edge_index.append([nbr_idx, i])

        # from neighbors to neighbors in both directions
        for i in range(num_query_nodes):
            neighbor_indices = list(
                range(num_query_nodes + (num_neighbors * i), num_query_nodes + (num_neighbors * (i + 1))))
            for j in neighbor_indices[:-1]:
                for k in neighbor_indices[j+1:]:
                    edge_index.append([j, k])
                    edge_index.append([k, j])

        edge_index = torch.Tensor(edge_index).to(dtype=torch.int64).to(query_encoding.device).transpose(0,1)

        # Loop through the GAT conv layers and update nodes and edges
        for gat_conv_layer in self.gat:
            #node_features, _, attention_weights = gat_conv_layer(x=node_features,
            #                                                     edge_index=edge_index, return_attention_weights=True).relu() # newer version
            node_features = gat_conv_layer(x=node_features, edge_index=edge_index).relu()

        # Concatenate node features to get edge features and pass through an MLP
        edge_features = torch.zeros((num_query_nodes * num_neighbors,
                                    node_features.shape[1]*2)).to(node_features.dtype).to(node_features.device)

        edge_atts = torch.zeros(num_query_nodes, num_neighbors).to(node_features.dtype).to(node_features.device)
        edge_count = 0
        for i in range(num_query_nodes):
            query_feature = node_features[i, :]
            neighbor_indices = list(
                range(num_query_nodes + (num_neighbors * i), num_query_nodes + (num_neighbors * (i + 1))))
            neighbor_features = node_features[neighbor_indices, :]
            for j in range(num_neighbors):
                edge_features[edge_count, :] = torch.cat((query_feature, neighbor_features[j]), dim=0)
                edge_atts[i, j] = torch.dot(query_feature, neighbor_features[j])
                edge_count += 1

        edge_atts = torch.nn.functional.softmax(edge_atts, dim=1)
        edge_features = self.edge_feature_updator(edge_features)

        # Regress the relative poses
        rel_x = self.x_regressor(edge_features)
        rel_q = self.q_regressor(edge_features)

        # Compute candidate absolute poses
        abs_pose = None
        if neighbor_poses is not None:
            abs_pose = compute_abs_pose(rel_x, rel_q, edge_atts, neighbor_poses)

        # Compute the absolute pose through affine combination based on the attention weights
        return {"rel_pose":torch.cat((rel_x, rel_q), dim=1), "edge_atts":edge_atts, "pose":abs_pose}











