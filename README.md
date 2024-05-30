# CSAI-Thesis-Marogna
This repository contains all the code used for the thesis "Using Diffusion Models to Simulate Fluorodeoxyglucose Uptake from Computed Tomography Scans" written by Francesca Luigia Marogna submitted in partial fulfillment of the requirements for the degree of Master of Science in Cognitive Science &amp; Artificial Intelligence at the Tilburg Univeristy.

For the preprocessing and cropping run the "cropping_PETCT.py" file with the following line:
- for cropping size 128:
python script.py --crop_size 128 --random_int 25 --img_path_pet PET_crop_128 --img_path_ct CT_crop_128 --img_path_mask mask_crop_128 --base_dir /home/u401763/Data/Size_128

- for cropping size 64:
python script.py --crop_size 64 --random_int 12 --img_path_pet PET_crop_64 --img_path_ct CT_crop_64 --img_path_mask mask_crop_64 --base_dir /home/u401763/Data/Size_64
