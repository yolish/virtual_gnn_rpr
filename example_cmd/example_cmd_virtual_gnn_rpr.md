# Virtual RPR with GNN and PAE
## Training multi-scene virtual RPR for the Cambridge Landmarks dataset  
```
main_learn_gnn_rpr_with_pae.py.py
ems-transposenet
train 
models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/cambridge_four_scenes.csv
CambridgeLandmarks_gnn_rpr_config.json
pretrained_models/ems_transposenet_cambridge_pretrained_finetuned.pth
pretrained_models/mstransformer_cambridge_pose_encoder.pth
```

## Testing multi-scene virtual RPR for the Cambridge Landmarks dataset  (KingsCollege scene)
```
main_learn_gnn_rpr_with_pae.py.py
ems-transposenet
test 
models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_gnn_rpr_config.json
pretrained_models/ems_transposenet_cambridge_pretrained_finetuned.pth
pretrained_models/mstransformer_cambridge_pose_encoder.pth
--ref_poses_file
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
--checkpoint_path
out/run_19_07_22_19_28_rpr_checkpoint-100.pth
```

## Training multi-scene virtual RPR for the 7Scenes dataset  
```
main_learn_gnn_rpr_with_pae.py.py
ems-transposenet
train 
models/backbones/efficient-net-b0.pth
/data/Datasets/7Scenes/
datasets/7Scenes/7scenes_all_scenes
7scenes_gnn_rpr_config.json
pretrained_models/ems_transposenet_7scenes_pretrained.pth
pretrained_models/mstransformer_7scenes_pose_encoder.pth
```

## Testing multi-scene virtual RPR for the 7Scenes dataset  
```
main_learn_gnn_rpr_with_pae.py.py
ems-transposenet
train 
models/backbones/efficient-net-b0.pth
/data/Datasets/7Scenes/
datasets/7Scenes/abs_7scenes_pose.csv_chess_test.csv
7scenes_gnn_rpr_config.json
pretrained_models/ems_transposenet_7scenes_pretrained.pth
pretrained_models/mstransformer_7scenes_pose_encoder.pth
--ref_poses_file
datasets/7Scenes/abs_7scenes_pose.csv_chess_train.csv
--checkpoint_path
'''
