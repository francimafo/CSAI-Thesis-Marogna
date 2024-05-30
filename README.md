# CSAI Thesis Marogna
This repository contains all the code used for the thesis "Using Diffusion Models to Simulate Fluorodeoxyglucose Uptake from Computed Tomography Scans" written by Francesca Luigia Marogna submitted in partial fulfillment of the requirements for the degree of Master of Science in Cognitive Science &amp; Artificial Intelligence at the Tilburg Univeristy.

## Preprocessing

For the preprocessing and cropping run the `cropping_PETCT.py` file with the following line:
- for cropping size 128:
```commandline
python cropping_PETCT.py --crop_size 128 --random_int 25 --img_path_pet PET_crop_128 --img_path_ct CT_crop_128 --img_path_mask mask_crop_128 --base_dir /path/to/cropped_data_128
```
- for cropping size 64:
```commandline
python cropping_PETCT.py --crop_size 64 --random_int 12 --img_path_pet PET_crop_64 --img_path_ct CT_crop_64 --img_path_mask mask_crop_64 --base_dir /path/to/cropped_data_128
```

## Train-val-test splitting

For the train-cal-test splitting run the `Splitting_train_test_val.py` file with the following line:
```commandline
python Splitting_train_test_val.py --ct_dataset_path /path/to/ct_crop--pet_dataset_path /path/to/pet_crop --mask_dataset_path /path/to/mask_crop
```
