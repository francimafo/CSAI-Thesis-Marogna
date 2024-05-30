import os
import shutil

'''
File to split from the PET_CT dataset the PET, CT, and mask into different folders.
Files are then put in the data_clean folder into image_pet, mask, image_ct folders.
'''

def main():
    # Replace this path with the actual path to your data directory
    DATA = "/home/llong/PET_CT"

    for scan_folder in os.listdir(DATA):
        if os.path.splitext(scan_folder)[1] == ".csv":
            continue

        id = scan_folder[6:]
        scan = os.path.join(DATA, scan_folder)
        scan_files = os.listdir(scan)

        for i in range(0, len(scan_files)):
            scans = os.path.join(scan, scan_files[i])
            ct = os.path.join(scans, "CTres.nii.gz")
            annot = os.path.join(scans, "SEG.nii.gz")
            pet = os.path.join(scans, "PET.nii.gz")

            if not os.path.isfile(ct) or not os.path.isfile(annot) or not os.path.isfile(pet):
                continue

            try:
                # Move ct
                if len(scan_files) > 1:
                    ct_dest = "/home/u401763/data_clean/image_ct/" + id + f"_{i}.nii.gz"
                else:
                    ct_dest = "/home/u401763/data_clean/image_ct/" + id + ".nii.gz"
                shutil.copy(ct, ct_dest)

                # Move annot
                if len(scan_files) > 1:
                    annot_dest = "/home/u401763/data_clean/mask/" + id + f"_{i}.nii.gz"
                else:
                    annot_dest = "/home/u401763/data_clean/mask/" + id + ".nii.gz"
                shutil.copy(annot, annot_dest)

                # Move pet
                if len(scan_files) > 1:
                    pet_dest = "/home/u401763/data_clean/image_pet/" + id + f"_{i}.nii.gz"
                else:
                    pet_dest = "/home/u401763/data_clean/image_pet/" + id + ".nii.gz"
                shutil.copy(pet, pet_dest)
            except Exception as e:
                print(f"An error occurred: {e}")

########### RUN ###########

if __name__ == "__main__":
    main()
