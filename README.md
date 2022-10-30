This is the code repository implementing the paper "Morig: Motion-aware rigging of character meshes from point clouds".

## Setup
The project is developed on Ubuntu 20.04 with cuda11.3.

```
conda env create -f environment.yml
conda activate morig
```

## Datasets
Download the processed datasets we used from the following links:
1. [ModelsResources](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EeiBvoeAJdVPl3Sx9xKHhisBcBulUu63IZnOaXJ0ZtfEqQ?e=C36xcb) (16.9G)
2. [DeformingThings4D](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EW_2pbf8LNRNhofmgf-XTasB8VyU8I-r3F1bv1qU9lmhIQ?e=aQNUBu) (10.5G)


## Testing & Evaluation

### Pretrained models
Download our pretrained models from [here](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EU011m-IkQhBkeItRXuKJzoBlwAy2GG2DhZERFCOfn8NVg?e=pHhLFL).

### Demo
We provide a demo script that concatenates all the steps, as a reference of the whole pipeline.
```
[To do]
```

### Evaluate on ModelsResources dataset:
1. Output shifted points and their attention. Remember to change the dataset and the output folder in each command to your preference.
```
python -u training/train_rig.py --arch="jointnet_motion" -e --resume ="checkpoints/jointnet_motion/model_best.pth.tar" --train_folder="DATASET_PATH/ModelsResources/train/" --val_folder="DATASET_PATH/ModelsResources/val/" --test_folder="DATASET_PATH/ModelsResources/test/" --output_folder="results/our_results"
python -u training/train_rig.py --arch="masknet_motion" -e --resume ="checkpoints/masknet_motion/model_best.pth.tar" --train_folder="DATASET_PATH/ModelsResources/train/" --val_folder="DATASET_PATH/ModelsResources/val/" --test_folder="DATASET_PATH/ModelsResources/test/" --output_folder="results/our_results"
```

2. Extract joints. 
Change the dataset and the result folders at line 49-55 in evaluate/eval_rigging.py.
We have set the optimal hyper-parameters by default.
```
python -u evaluate/eval_rigging.py
```

3. Connect joints to form skeletons using ```pred_skel_func``` in evaluate/joint2rig.py.
Then, predict skinning weights to form rigs using ```pred_rig_func``` in evaluate/joint2rig.py.
Remember to change dataset_folder in each function.

5. Animate characters according to partial point cloud sequences. Remember to set the dataset and the results folders at line 279-282.
```
python -u evaluate/eval_tracking.py
```

I put our results [here](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/Ea7eMXkBdlRDrrtkB3RKYRUBC_UiuPpHGg0eahx_sVipRw?e=12YuVO) for your reference.

## Training
One can run the following steps to train all the networks.

1. Train a correspondence module with discrete frames on ModelsResources dataset.
```
python -u training/train_corr_pose.py \
--train_folder="DATASET_PATH/ModelsReources/train/" \
--val_folder="DATASET_PATH/ModelsReources/val/" \
--test_folder="DATASET_PATH/ModelsReources/test/" 
--train_batch=8 --test_batch=8 \
--logdir="logs/corr_p_mr" \ 
--checkpoint="checkpoints/corr_p_mr" \ 
--num_workers=4 --lr=1e-3 \
--vis_branch_start_epoch=100 --schedule 200 \
--epochs=300 --dataset="modelsresource"
```
2. (After 1) Train a deformation module with discrete frames on ModelsResources dataset
```
python -u training/train_deform_pose.py \
--train_folder="DATASET_PATH/ModelsReources/train/" \
--val_folder="DATASET_PATH/ModelsReources/val/" \
--test_folder="DATASET_PATH/ModelsReources/test/" \
--train_batch=6 --test_batch=6 \
--logdir="logs/deform_p_mr" \
--checkpoint="checkpoints/deform_p_mr" \
--init_extractor="checkpoints/corr_p_mr/model_best.pth.tar" \
--num_workers=4 --lr=1e-4 --epochs=150 --schedule 60 120 \
--dataset="modelsresource"
```
3. (After 1 and 2) Train a joint prediction module. We provided the predicted deformation (folder "pred_flow") in our preprocessed dataset. 
You need to output predicted deformation if you use different data:
```
python -u training/train_rig.py \
--arch="jointnet_motion" \
--train_folder="DATASET_PATH/ModelsReources/train/" \
--val_folder="DATASET_PATH/ModelsReources/val/" \
--test_folder="DATASET_PATH/ModelsReources/test/" \
--train_batch=4 --test_batch=4 \
--logdir="logs/jointnet_motion" \
--checkpoint="checkpoints/jointnet_motion" \
--lr=5e-4 --schedule 40 80 --epochs=120
```
4. (After 1 and 2) Similar to 3, train an attention prediction module.
```
python -u training/train_rig.py \
--arch="masknet_motion" \
--train_folder="DATASET_PATH/ModelsReources/train/" \
--val_folder="DATASET_PATH/ModelsReources/val/" \
--test_folder="DATASET_PATH/ModelsReources/test/" \
--train_batch=4 --test_batch=4 \
--logdir="logs/masknet_motion" \
--checkpoint="checkpoints/masknet_motion" \
--lr=5e-4 --schedule 50 --epochs=100
```

5. (After 1 and 2) Similar to 3 and 4, train a skinning prediction module.
```
python -u training/train_skin.py \
--arch="skinnet_motion" \
--train_folder="DATASET_PATH/ModelsReources/train/" \
--val_folder="DATASET_PATH/ModelsReources/val/" \
--test_folder="DATASET_PATH/ModelsReources/test/" \
--train_batch=4 --test_batch=4 \
--logdir="logs/skin_motion" \
--checkpoint="checkpoints/skin_motion" \
--loss_cont="infonce" \
-epochs=100
```

6. To animate the rigged character based on point cloud sequence, we train a correspondence module and a 
deformation module with sequential frames on ModelsResources dataset. This can be achieved by simply adding "--sequential_frame":
```
python -u training/train_corr_pose.py \
--train_folder="DATASET_PATH/ModelsReources/train/" \
--val_folder="DATASET_PATH/ModelsReources/val/" \
--test_folder="DATASET_PATH/ModelsReources/test/" 
--train_batch=8 --test_batch=8 \
--logdir="logs/corr_p_mr_seq" \ 
--checkpoint="checkpoints/corr_p_mr_seq" \ 
--num_workers=4 --lr=1e-3 \
--vis_branch_start_epoch=100 --schedule 200 \
--epochs=300 --dataset="modelsresource" --sequential_frame
```
```
python -u training/train_deform_pose.py \
--train_folder="DATASET_PATH/ModelsReources/train/" \
--val_folder="DATASET_PATH/ModelsReources/val/" \
--test_folder="DATASET_PATH/ModelsReources/test/" \
--train_batch=6 --test_batch=6 \
--logdir="logs/deform_p_mr_seq" \
--checkpoint="checkpoints/deform_p_mr_seq" \
--init_extractor="checkpoints/corr_p_mr_seq/model_best.pth.tar" \
--num_workers=4 --lr=1e-4 --epochs=150 --schedule 60 120 \
--dataset="modelsresource" --sequential_frame
```

7. To better generalize to real motion, we finetune correspondence and deformation modules on DeformingThings4D dataset. 
This can be achieved by setting "--dataset" to "deformingthings".
```
python -u training/train_corr_pose.py \
--train_folder="DATASET_PATH/DeformingThings4D/train/" \
--val_folder="DATASET_PATH/DeformingThings4D/val/" \
--test_folder="DATASET_PATH/DeformingThings4D/test/" \
--train_batch=8 --test_batch=8 \
--logdir="logs/corr_p_dt_seq" \
--checkpoint="checkpoints/corr_p_dt_seq" \
--resume="checkpoints/corr_p_mr_seq/model_best.pth.tar"
--num_workers=4 --lr=1e-3 \
--vis_branch_start_epoch=100 --schedule 200 \
--epochs=300 --dataset="deformingthings" --sequential_frame
```
```
python -u training/train_deform_pose.py \
--train_folder="DATASET_PATH/DeformingThings4D/train/" \
--val_folder="DATASET_PATH/DeformingThings4D/val/" \
--test_folder="DATASET_PATH/DeformingThings4D/test/" \
--train_batch=6 --test_batch=6 \
--logdir="logs/deform_p_dt_seq" \
--checkpoint="checkpoints/deform_p_dt_seq" \
--init_extractor="checkpoints/corr_p_dt_seq/model_best.pth.tar"
--num_workers=4 --lr=1e-4 --epochs=150 --schedule 60 120 \
--dataset="deformingthings" --sequential_frame
```

8. When the shape of target mesh and the captured point cloud are different, 
we first deform the shape of the mesh to fit the point cloud. 
This can be achieved by the same correspondence and deformation module architecture trained on data with different shape (train_deform/val_deform/test_deform):
```
python -u training/train_corr_shape.py \
--train_folder="DATASET_PATH/ModelsReources/train_deform/" \
--val_folder="DATASET_PATH/ModelsReources/val_deform/" \
--test_folder="DATASET_PATH/ModelsReources/test_deform/" 
--train_batch=8 --test_batch=8 \
--logdir="logs/corr_s_mr" \ 
--checkpoint="checkpoints/corr_s_mr" \ 
--num_workers=4 --lr=1e-3 \
--vis_branch_start_epoch=100 --schedule 200 \
--epochs=300 --dataset="modelsresource"
```
```
python -u training/train_deform_shape.py \
--train_folder="DATASET_PATH/ModelsReources/train_deform/" \
--val_folder="DATASET_PATH/ModelsReources/val_deform/" \
--test_folder="DATASET_PATH/ModelsReources/test_deform/" \
--train_batch=6 --test_batch=6 \
--logdir="logs/deform_s_mr" \
--checkpoint="checkpoints/deform_s_mr" \
--init_extractor="checkpoints/corr_s_mr/model_best.pth.tar" \
--num_workers=4 --lr=1e-4 --epochs=150 --schedule 60 120 \
--dataset="modelsresource"
```
