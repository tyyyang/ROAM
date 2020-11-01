## ROAM: Recurrently Optimizing Tracking Model
### Introduction
This is the PyTorch implementation of our [ROAM](https://arxiv.org/pdf/1907.12006.pdf) tracker published in CVPR, 2020. Detailed comparision results can be found in the author's [webpage](https://tianyu-yang.com)

### Prerequisites

* Python 3.5 or higher
* PyTorch 1.4.0 or higher

### Path setting
Set proper `root_dir` in `config.py` accordingly in order to proceed the following step. Make sure that you place the tracking data properly according to your path setting.

### Training
1. Download the ILSRVC data from the official website and extract it to proper place according to the path in `config.py`. Pretrained `vgg-16.mat` file can be download from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat)
2. Then run the `python3 make_vid_info.py` in to build the meta data file for  ILSVRC data.
3. Run: 
```
python3 experiment.py \
  --mGPUs \
  --epochs 20 \
  --bs [BATCH_SIZE] \
  --nw [NUM_WORKERS] \
  --lr_mi 1e-6 \
  --lr_mo 1e-3
``` 
to train the model. Note we train our model on a 4-GPUs machine with `BATCH_SIZE=16`

### Tracking Demo
After training, you can run `python3 demo.py` to test our tracker.

### Citing ROAM
If you find the code is helpful, please cite
```
@inproceedings{Yang2020cvpr,
	author = {Yang, Tianyu and Xu, Pengfei and Hu, Runbo and Chai, Hua and Chan, Antoni B},
	booktitle = {CVPR},
	title = {{ROAM: Recurrently Optimizing Tracking Model}},
	year = {2020}
}
```
