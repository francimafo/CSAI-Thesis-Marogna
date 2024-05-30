import os
import random
import shutil
import argparse


def split_and_move_images(dataset_path, patient_images, split_patients, split_path):
    os.makedirs(split_path, exist_ok=True)
    for patient_id in split_patients:
        for image_file in patient_images[patient_id]:
            source_path = os.path.join(dataset_path, image_file)
            destination_path = os.path.join(split_path, image_file)
            shutil.move(source_path, destination_path)


def main(ct_dataset_path, pet_dataset_path, mask_dataset_path):
    ## CT
    # Get the list of all PNG files in the CT dataset directory
    ct_png_files = [file for file in os.listdir(ct_dataset_path) if file.endswith('.png')]

    # Create a dictionary to store patient IDs and their corresponding images
    ct_patient_images = {}
    for file in ct_png_files:
        patient_id = file.split('_')[1]
        if patient_id not in ct_patient_images:
            ct_patient_images[patient_id] = []
        ct_patient_images[patient_id].append(file)

    # Get a list of unique patient IDs
    unique_patient_ids = list(ct_patient_images.keys())

    # Shuffle the unique patient IDs
    random.shuffle(unique_patient_ids)

    # Calculate the split sizes
    total_patients = len(unique_patient_ids)
    train_size = int(0.8 * total_patients)
    val_size = int(0.1 * total_patients)
    test_size = total_patients - train_size - val_size

    # Split the patient IDs
    train_patients = unique_patient_ids[:train_size]
    val_patients = unique_patient_ids[train_size:train_size + val_size]
    test_patients = unique_patient_ids[train_size + val_size:]

    # Create directories for train, validation, and test sets
    train_path = os.path.join(ct_dataset_path, 'train')
    val_path = os.path.join(ct_dataset_path, 'val')
    test_path = os.path.join(ct_dataset_path, 'test')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Move the CT files to the corresponding sets
    split_and_move_images(ct_dataset_path, ct_patient_images, train_patients, train_path)
    split_and_move_images(ct_dataset_path, ct_patient_images, val_patients, val_path)
    split_and_move_images(ct_dataset_path, ct_patient_images, test_patients, test_path)

    print("Finished splitting CT images")

    ### PET
    # Get the list of all PNG files in the PET dataset directory
    pet_png_files = [file for file in os.listdir(pet_dataset_path) if file.endswith('.png')]

    # Create a dictionary to store patient IDs and their corresponding images for PET
    pet_patient_images = {}
    for file in pet_png_files:
        patient_id = file.split('_')[1]
        if patient_id not in pet_patient_images:
            pet_patient_images[patient_id] = []
        pet_patient_images[patient_id].append(file)

    # Move the PET files to the corresponding sets
    split_and_move_images(pet_dataset_path, pet_patient_images, train_patients, os.path.join(pet_dataset_path, 'train'))
    split_and_move_images(pet_dataset_path, pet_patient_images, val_patients, os.path.join(pet_dataset_path, 'val'))
    split_and_move_images(pet_dataset_path, pet_patient_images, test_patients, os.path.join(pet_dataset_path, 'test'))

    print("Finished splitting PET images")

    ### Mask
    # Get the list of all PNG files in the mask dataset directory
    mask_png_files = [file for file in os.listdir(mask_dataset_path) if file.endswith('.png')]

    # Create a dictionary to store patient IDs and their corresponding images for mask
    mask_patient_images = {}
    for file in mask_png_files:
        patient_id = file.split('_')[1]
        if patient_id not in mask_patient_images:
            mask_patient_images[patient_id] = []
        mask_patient_images[patient_id].append(file)

    # Move the mask files to the corresponding sets
    split_and_move_images(mask_dataset_path, mask_patient_images, train_patients,
                          os.path.join(mask_dataset_path, 'train'))
    print("Training set done")
    split_and_move_images(mask_dataset_path, mask_patient_images, val_patients, os.path.join(mask_dataset_path, 'val'))
    print("Validation set done")
    split_and_move_images(mask_dataset_path, mask_patient_images, test_patients,
                          os.path.join(mask_dataset_path, 'test'))
    print("Test set done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--ct_dataset_path', type=str, required=True, help='Path to the CT dataset')
    parser.add_argument('--pet_dataset_path', type=str, required=True, help='Path to the PET dataset')
    parser.add_argument('--mask_dataset_path', type=str, required=True, help='Path to the mask dataset')

    args = parser.parse_args()

    main(args.ct_dataset_path, args.pet_dataset_path, args.mask_dataset_path)
