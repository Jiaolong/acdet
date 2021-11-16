# ACDet: Attentive Cross-view Fusion for LiDAR-based 3D Object Detection


## Getting Started

This code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## Models

Pre-trained models can be downloaded from [link](https://drive.google.com/drive/folders/1eVjeOXTxn_ymEj8-Va5WMhsq19s2jjiJ?usp=sharing)

## Training & Testing

An example configuration file is located at `tools/cfgs/kitti_models/cyv_det_metav6_fuse2_raw_transformer_mask.yaml`.

### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
`
