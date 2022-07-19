"""
Entry point for learning GNN RPR with PAE
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
from models.pose_encoder import PoseEncoder, MultiSCenePoseEncoder
from models.RPGNN import RPGNN, compute_rel_pose


def get_ref_pose(query_poses, db_poses, start_index, k, sample=False):
    ref_poses = np.zeros((query_poses.shape[0], k, 7))
    for i, p in enumerate(query_poses):
        dist_x = np.linalg.norm(p[:3] - db_poses[:, :3], axis=1)
        dist_x = dist_x / np.max(dist_x)
        dist_q = np.linalg.norm(p[3:] - db_poses[:, 3:], axis=1)
        dist_q = dist_q / np.max(dist_q)
        sorted = np.argsort(dist_x + dist_q)
        if sample:
            indices = list(range(1, k*3+1))
            sampled_indices = np.random.choice(indices, size=k, replace=False)
            ref_poses[i, 0:k, :] = db_poses[sorted[sampled_indices]]
        else:
            ref_poses[i, 0:k, :] = db_poses[sorted[start_index:(k+start_index)]]
    return ref_poses

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("apr_checkpoint_path",
                            help="path to a pre-trained apr model (should match the model indicated in model_name")
    arg_parser.add_argument("encoder_checkpoint_path",
                            help="path to a pre-trained encoder model (should match the APR model")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained RPR model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--ref_poses_file", help="path to file with train poses")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    apr = get_model(args.model_name, args.backbone_path, config).to(device)
    apr.load_state_dict(torch.load(args.apr_checkpoint_path, map_location=device_id))
    logging.info("Initializing from checkpoint: {}".format(args.apr_checkpoint_path))
    apr.eval()

    pose_encoder = MultiSCenePoseEncoder(config.get("hidden_dim")).to(device)
    pose_encoder.load_state_dict(torch.load(args.encoder_checkpoint_path, map_location=device_id))
    logging.info("Initializing encoder from checkpoint: {}".format(args.encoder_checkpoint_path))
    pose_encoder.eval()

    # Create the GNN RPR model
    gnn_rpr = RPGNN(config)
    gnn_rpr.to(device)

    # Load the checkpoint if needed
    if args.checkpoint_path:
        gnn_rpr.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    num_neighbors = int(config.get("num_neighbors"))

    if args.mode == 'train':
        # Set to train mode
        gnn_rpr.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(gnn_rpr.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                minibatch['scene'] = None  # avoid using ground-truth scene during prediction
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                with torch.no_grad():
                    # estimate pose and scene
                    res = apr(minibatch)
                    est_pose = res.get('pose')
                    scene_dist = res.get('scene_log_distr')
                    scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32).unsqueeze(1)

                    # Encode the pose
                    latent_x, latent_q = pose_encoder(est_pose, scene)

                    # Get the k closest poses and encode them
                    closest_poses = get_ref_pose(est_pose.cpu().numpy(), dataset.poses, 1, num_neighbors, sample=True)
                    closest_poses = torch.Tensor(closest_poses).to(device).view(-1, 7)
                    ref_latent_x, ref_latent_q = pose_encoder(closest_poses, scene.repeat(num_neighbors, 1))

                    query_encoding = torch.cat((latent_x, latent_q), dim=1)
                    neighbors_encoding = torch.cat((ref_latent_x, ref_latent_q), dim=1)

                # Use RPR to estimate the relative poses and the absolute pose
                # Zero the gradients
                optim.zero_grad()

                res = gnn_rpr(query_encoding, neighbors_encoding, closest_poses)

                gt_rel_pose = torch.zeros_like(closest_poses)
                rel_pose_count = 0
                for q_idx in range(batch_size):
                    nbr_indices = list(range(q_idx * num_neighbors, (q_idx + 1) * num_neighbors))
                    for nbr_idx in range(num_neighbors):
                        gt_rel_pose[rel_pose_count] = compute_rel_pose(gt_pose[q_idx].unsqueeze(0),
                                                                       closest_poses[nbr_indices[nbr_idx]].unsqueeze(0))
                        rel_pose_count += 1

                # Pose loss
                est_rel_pose = res['rel_pose']
                est_abs_pose = res['pose']
                criterion = pose_loss(est_rel_pose, gt_rel_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_abs_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(gnn_rpr.state_dict(), checkpoint_prefix + '_rpr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(gnn_rpr.state_dict(), checkpoint_prefix + '_rpr_final.pth'.format(epoch))

    else: # Test
        # Set to eval mode
        gnn_rpr.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)
        ref_poses = CameraPoseDataset(args.dataset_path, args.ref_poses_file, None).poses

        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats_encoder = np.zeros(len(dataloader.dataset))
        stats_retrieval = np.zeros(len(dataloader.dataset))
        stats_decoder = np.zeros(len(dataloader.dataset))
        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None # avoid using ground-truth scene during prediction

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # estimate pose and scene
                res = apr(minibatch)
                est_pose = res.get('pose')
                scene_dist = res.get('scene_log_distr')
                scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32).unsqueeze(1)

                # Encode the pose
                latent_x, latent_q = pose_encoder(est_pose, scene)

                # Get the k closest poses and encode them
                tic = time.time()
                closest_poses = get_ref_pose(est_pose.cpu().numpy(), ref_poses, 0, num_neighbors)
                closest_poses = torch.Tensor(closest_poses).to(device).view(-1, 7)
                ref_latent_x, ref_latent_q = pose_encoder(closest_poses, scene.repeat(num_neighbors, 1))

                query_encoding = torch.cat((latent_x, latent_q), dim=1)
                neighbors_encoding = torch.cat((ref_latent_x, ref_latent_q), dim=1)

                # Use RPR to estimate the relative poses and the absolute pose
                res = gnn_rpr(query_encoding, neighbors_encoding, closest_poses)
                est_pose = res['pose']
                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))







