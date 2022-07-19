## Virtual Relative Pose Regression with Camera Pose Auto-Encoders (PAEs) and Graph Neural Networks (GNNs)

### Repository Overview 

This code implements training and testing of multi-scene *virtual* pose regresors (RPRs) with a GNN architecture and [PAE encodings](https://github.com/yolish/camera-pose-auto-encoders).
To learn more about PAEs, please refer to our ECCV22 paper and repository about [camera pose auto-encoders](https://github.com/yolish/camera-pose-auto-encoders) 

### Prerequisites

In order to run this repository you will need:
1. Python3 (tested with Python 3.7.7)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Use torch==1.4.0, torchvision==0.5.0
1. Pytorch geometric - see its [installation guide]()  
1. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset
1. You can also download pre-trained models to reproduce reported results (see below)
1. For a quick set up you can run: pip install -r requirments.txt and then install pytorch geometric 

### Usage
### Pretrained PAEs and APRs
You can download pre-trained PAEs and APRs required for training virtual relative pose regression. In addition
You may also download our pre-trained RPRs for testing purposes. All models are available from the table below.
| Model (Linked) | Description | 
--- | ---
| Multi-scene APR models ||
[MS-Transformer](https://drive.google.com/file/d/1ZEIKQSbZmkSnJwETjACvMbs5OeCn7f3q/view?usp=sharing) | Multi-scene APR, CambridgeLandmarks dataset|
[MS-Transformer](https://drive.google.com/file/d/1Ryn5oQ0zRV_3KVORzMAk99cP0fY2ff85/view?usp=sharing) | Multi-scene APR, 7Scenes dataset|
| Camera Pose Auto-Encoders||
[Auto-Encoder for MS-Transformer](https://drive.google.com/file/d/1rshdruRQcZYMIRI9lTY_U981cJsohauI/view?usp=sharing) | Auto-Encoder for a multi-scene APR, CambridgeLandmarks dataset|
[Auto-Encoder for MS-Transformer](https://drive.google.com/file/d/1hGcII8D0G24DBGXh3aLohCubAmfN9Rc7/view?usp=sharing) | Auto-Encoder for a multi-scene APR, 7Scenes dataset|
| Multi-scene virtual RPR models | |
[Virtual RPR] () | Virtual RPR for the CambrdidgeLandmarks dataset |
[Virtual RPR] () | Virtual RPR for the 7Scenes dataset |

### Training and Testing of multi-scene virtual RPRs
The entry point for training and testing APRs is the ```main_learn_gnn_rpr_with_pae.py``` script in the root directory
See ```example_cmd/example_cmd_virtual_gnn_rpr.md``` for example command lines.
