# CSAI Thesis Marogna
This repository contains all the code used for the thesis "Using Diffusion Models to Simulate Fluorodeoxyglucose Uptake from Computed Tomography Scans" written by Francesca Luigia Marogna submitted in partial fulfillment of the requirements for the degree of Master of Science in Cognitive Science &amp; Artificial Intelligence at the Tilburg Univeristy.

For ease of use, the environment used for this project is provided as `cl_env.yml`.
## Preprocessing

For the preprocessing and cropping run the `Preprocessing/cropping_PETCT.py` file with the following line:
- for cropping size 128:
```commandline
python cropping_PETCT.py --crop_size 128 --random_int 25 --img_path_pet PET_crop_128 --img_path_ct CT_crop_128 --img_path_mask mask_crop_128 --base_dir /path/to/cropped_data_128
```
- for cropping size 64:
```commandline
python cropping_PETCT.py --crop_size 64 --random_int 12 --img_path_pet PET_crop_64 --img_path_ct CT_crop_64 --img_path_mask mask_crop_64 --base_dir /path/to/cropped_data_128
```

## Train-val-test splitting

For the train-cal-test splitting run the `Preprocessing/Splitting_train_test_val.py` file with the following line:
```commandline
python Splitting_train_test_val.py --ct_dataset_path /path/to/ct_crop--pet_dataset_path /path/to/pet_crop --mask_dataset_path /path/to/mask_crop
```

## Improved Diffusion
In order to run the DDIB models, we first need to create pre-trained models for CT and PET, for both size 64 and 128. We use the code from https://github.com/openai/improved-diffusion.git and make some modifications in order for the code to work for the project at hand. This author recommends to play around with the parameters, epecially the diffusion steps. The default is 4000. The following parameters are used in this project:
```commandline
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3" DIFFUSION_FLAGS="--diffusion_steps 400 --noise_schedule linear" TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
```
To run the models run the following line:
```commandline
mpiexec --mca opal_cuda_support 1 python improved-diffusion/scripts/image_train.py --data_dir /path/to/cropped_CT/train --microbatch 1 --log_dir /path/to/save/models $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS 
```

# DDIB
In order to run the DDIB models, run the following line:
```commandline
mpiexec --mca opal_cuda_support 1 python ddib/scripts/CTPET_translation.py
```
Remember to add the various arguments defining what your paths are.
