import os
import glob
import random
import argparse
import numpy as np
import nibabel as nib
from skimage import filters
from matplotlib import image
from skimage import transform
from skimage.measure import label, regionprops

def load_data(original_path_ct, original_path_pet, original_path_mask, label_path):
    patient_number = os.path.basename(label_path).split(".")[0] # Retrieving the patient number
    # Loading the NiFTI files
    original_data_ct = nib.load(os.path.join(original_path_ct, patient_number + ".nii.gz")).get_fdata()
    original_data_pet = nib.load(os.path.join(original_path_pet, patient_number + ".nii.gz")).get_fdata()
    original_data_mask = nib.load(os.path.join(original_path_mask, patient_number + ".nii.gz")).get_fdata()
    label_data = nib.load(label_path).get_fdata()
    return original_data_ct, original_data_pet, original_data_mask, label_data, patient_number

# Normalizing the images: minmax
def NormalizeData_minmax(image):
    max = np.max(image)
    min = np.min(image)
    final_image = (image - min) / (max - min)
    return final_image

# Code to find lesions and crop around the centroid using regionprops
def regionprops_and_crop(image_data_ct, image_data_pet, image_data_mask, label_data, patient_number, crop_size, randomInt, img_path_pet, img_path_ct, img_path_mask, base_dir):
    crop_xy = int(crop_size // 2) # Crop size either being 64 or 128
    for i in range(label_data.shape[2]):
        threshold = filters.threshold_otsu(label_data[:, :, i])
        mask = label_data[:, :, i] > threshold
        slice_label = label(mask)
        regions = regionprops(slice_label)

        for props in regions:
            x0_or, y0_or = props.centroid
            randomIntX = random.randint(-randomInt, randomInt)
            randomIntY = random.randint(-randomInt, randomInt)
            x0 = int(x0_or + randomIntX)
            y0 = int(y0_or + randomIntY)
            z0 = i
            if 50 < x0 < 710 and 50 < y0 < 710: # Adjust these values based on your area of interest
                slice_label = label_data[:, :, z0]
                slice_im_ct = image_data_ct[:, :, i]
                slice_im_pet = image_data_pet[:, :, i]
                slice_im_mask = image_data_mask[:, :, i]
                crop_label = slice_label[(x0 - crop_xy):(x0 + crop_xy), (y0 - crop_xy):(y0 + crop_xy)]

                crop_im_ct = slice_im_ct[(x0 - crop_xy):(x0 + crop_xy), (y0 - crop_xy):(y0 + crop_xy)]
                crop_im_pet = slice_im_pet[(x0 - crop_xy):(x0 + crop_xy), (y0 - crop_xy):(y0 + crop_xy)]
                crop_im_mask = slice_im_mask[(x0 - crop_xy):(x0 + crop_xy), (y0 - crop_xy):(y0 + crop_xy)]

                # Check and create directories if they don't exist
                save_img_path_pet = os.path.join(base_dir, img_path_pet)
                save_img_path_ct = os.path.join(base_dir, img_path_ct)
                save_img_path_mask = os.path.join(base_dir, img_path_mask)

                os.makedirs(save_img_path_ct, exist_ok=True)
                os.makedirs(save_img_path_pet, exist_ok=True)
                os.makedirs(save_img_path_mask, exist_ok=True)

                if ((crop_im_ct.shape == (crop_size, crop_size) and crop_label.sum() > 0)
                        and (crop_im_pet.shape == (crop_size, crop_size) and crop_label.sum() > 0)
                        and (crop_im_mask.shape == (crop_size, crop_size) and crop_label.sum() > 0)):

                    # Resize the cropped patches to 256x256 before saving
                    resized_crop_im_ct = transform.resize(crop_im_ct, (256, 256))
                    resized_crop_im_pet = transform.resize(crop_im_pet, (256, 256))
                    resized_crop_im_mask = transform.resize(crop_im_mask, (256, 256))

                    image.imsave(os.path.join(save_img_path_ct, f"lesion_{patient_number}_{x0}_{y0}_{z0}.png"), resized_crop_im_ct, cmap='gray')
                    print(f"lesion_{patient_number}_{x0}_{y0}_{z0}.png saved CT")
                    image.imsave(os.path.join(save_img_path_pet, f"lesion_{patient_number}_{x0}_{y0}_{z0}.png"), resized_crop_im_pet, cmap='gray')
                    print(f"lesion_{patient_number}_{x0}_{y0}_{z0}.png saved PET")
                    image.imsave(os.path.join(save_img_path_mask, f"lesion_{patient_number}_{x0}_{y0}_{z0}.png"), resized_crop_im_mask, cmap='gray')
                    print(f"lesion_{patient_number}_{x0}_{y0}_{z0}.png saved mask")

    return

def final_run_petct(label_path, original_path_ct, original_path_pet, original_path_mask, crop_size, randomInt, img_path_pet, img_path_ct, img_path_mask, base_dir):
    label_path_list = glob.glob(os.path.join(label_path, "*.nii.gz"))
    for i in range(len(label_path_list)):
        image_data_ct, image_data_pet, image_data_mask, label_data, patient_number = load_data(original_path_ct, original_path_pet, original_path_mask, label_path_list[i])
        image_data_ct = NormalizeData_minmax(image_data_ct)
        image_data_pet = NormalizeData_minmax(image_data_pet)
        image_data_mask = NormalizeData_minmax(image_data_mask)
        regionprops_and_crop(image_data_ct, image_data_pet, image_data_mask, label_data, patient_number, crop_size, randomInt, img_path_pet, img_path_ct, img_path_mask, base_dir)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--crop_size', type=int, choices=[64, 128], required=True, help='Crop size: 64 or 128')
    parser.add_argument('--random_int', type=int, required=True, help='Random integer range for centroid shifting')
    parser.add_argument('--img_path_pet', type=str, required=True, help='Path to save PET cropped images')
    parser.add_argument('--img_path_ct', type=str, required=True, help='Path to save CT cropped images')
    parser.add_argument('--img_path_mask', type=str, required=True, help='Path to save mask cropped images')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for saving images')

    args = parser.parse_args()

    # Paths for original images
    ct_path = '/path/to/image_ct'
    label_path = '/path/to/mask/'
    pet_path = '/path/to/image_pet'
    mask_path = '/path/to/mask'

    final_run_petct(label_path, ct_path, pet_path, mask_path, args.crop_size, args.random_int, args.img_path_pet, args.img_path_ct, args.img_path_mask, args.base_dir)
