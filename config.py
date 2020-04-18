# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

# path setting
root_dir = ''
feat_dir = root_dir+'/Data/Pre-trained/'
otb_dir = root_dir+'/Data/OTB/'
vot_dir = root_dir+'/Data/VOT/'
model_dir = './models/'
log_dir = './logs'
disp_inter = 100
lr_decay = 0.5
n_decay = 5

# backbone
feat_channels = 512

# data preprocessing
look_ahead = 5
time_step = 6
base_target_sz = 56
cell_sz = 4
search_scale = 5
offset_range = base_target_sz*(search_scale-1)/2
output_sigma_factor = 0.1
alpha = 0.2
mean = [123.68, 116.779, 103.939]
std = [1, 1, 1]

# meta initilizer
cf_channels = 64
base_filter_size = [21, 21]
filter_scale = 1.5
rand_scale_radius_cf = 1.6
rand_scale_radius_reg = 1.3
meta_lr_init = 1e-6

# meta optimizer
lstm_hidden_size = 20
lstm_layer_num = 2

# bbox regression
anchor_pos_thres = 0.6
anchor_pos_num = 32
aug_ratios_range = [0.8, 1.2]
aug_scales_range = [0.8, 1.2]
aug_init_ratios = [1, 0.8, 1.2]
aug_init_scales = [1, 0.8, 1.2]

# online tracking
score_penalty_factor = 0.05
motion_sigma_factor = 0.5
n_init_updates = 1
n_online_updates = 2
update_interval = 5
n_update_batch = 5
max_db_size = n_online_updates * n_update_batch
size_decay = 0.6
handle_drift = True
miss_thres = 0.1
reliable_thres = 0.6