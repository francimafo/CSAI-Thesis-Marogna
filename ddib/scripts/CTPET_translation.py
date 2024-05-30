import os
import sys
import csv
import torch
import argparse
import numpy as np
from skimage import io, color
from datetime import datetime
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from numpy import cov, trace, iscomplexobj
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    create_model_and_diffusion,
    args_to_dict
)

join = os.path.join
# Added this line to access the improved_diffusion module
sys.path.append('/path/to/Code/ddib')

def create_argparser():
    defaults = dict(
        image_size=256,
        batch_size=1,
        num_channels=128,
        num_res_blocks=3,
        num_heads=1,
        diffusion_steps=800,
        noise_schedule='linear',
        lr=1e-4,
        clip_denoised=False,
        num_samples=1,
        use_ddim=True,
        model_path="",
    )
    ori = model_and_diffusion_defaults()
    ori.update(defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, ori)
    return parser

if __name__ == "__main__":
    parser = create_argparser()
    parser.add_argument('--ct_dataset_path', type=str, required=True,
                        help='Path to the CT dataset')
    parser.add_argument('--pet_dataset_path', type=str, required=True,
                        help='Path to the PET dataset')
    parser.add_argument('--mask_dataset_path', type=str, required=True,
                        help='Path to the mask dataset')
    parser.add_argument('--models_path', type=str, required=True,
                        help='Path to the directory containing pre-trained models')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to log created ddib models. NEED TO BE inside the models_path')
    parser.add_argument('--ct_model_path', type=str, required=True,
                        help='Path to the CT pre-trained model')
    parser.add_argument('--pet_model_path', type=str, required=True,
                        help='Path to the PET pre-trained model')
    args = parser.parse_args()

    logger.log(f"args: {args}")

    # Defining the paths to the pre-trained models
    models_path = args.models_path

    petct_log = args.log_dir 
    logger.configure(dir=join(models_path, petct_log))
    code_folder = './'
    
    # load model
    def read_model_and_diffusion(args, model_path):
        """Reads the latest model from the given directory."""
    
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys()),
        )
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cuda"))
        model.to(dist_util.dev())
        model.eval()
        return model, diffusion
    
    # Frechet Inception Distance
    def fid(act1, act2):
        # Flatten the images to shape (num_pixels, num_channels)
        act1 = act1.reshape(-1, act1.shape[-1])
        act2 = act2.reshape(-1, act2.shape[-1])
    
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid_value = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid_value
    
    
    # Defining the paths to the pre-trained CT and reading the model
    ct_model_path = args.ct_model_path
    ct_model_path = join(models_path, ct_model_path)
    s_model, s_diffusion = read_model_and_diffusion(args, ct_model_path)
    
    # Defining the paths to the pre-trained PET and reading the model
    pet_model_path = args.pet_model_path
    pet_model_path = join(models_path, pet_model_path)
    t_model, t_diffusion = read_model_and_diffusion(args, pet_model_path)
    
    # Path to save the PET/CT diffusion model
    save_path = join(models_path, petct_log)
    
    # Paths to the datasets
    s_img_path = join(args.ct_dataset_path, 'CT_crop', 'train')
    pet_img_path = join(args.pet_dataset_path, 'PET_crop', 'train')
    mask_img_path = join(args.mask_dataset_path, 'mask_crop', 'train')
    names = sorted(os.listdir(s_img_path))
    
    # Pre-processing step
    def sample2img(sample):
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous().cpu().numpy()[0]
        return sample
    
    # Initialize the CSV file path outside the loop
    csv_file_path = join(save_path, 'progress.csv')
    
    # Open the CSV file in append mode ('a') outside the loop
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        # Write the header only once, before the loop starts
        writer.writerow(['Patient ID', 'SSIM_PET', 'PSNR_PET', 'FID_PET', 'SSIM_CT', 'PSNR_CT', 'FID_CT', 'Start Time', 'End Time', 'Time Taken'])
    
        # Iterate through names
        for name in names:
            # Record the start time
            start_time = datetime.now()
            print('starting time:', start_time)
            # CT data
            ct_data = io.imread(join(s_img_path, name))
            # Select the first three channels explicitly
            ct_data = ct_data[:, :, :3]
            s_np = ct_data / np.max(ct_data)
            s_np = (s_np - 0.5) * 2.0
            assert s_np.shape == (256, 256, 3), f"Shape error! Current shape: {ct_data.shape}"
            s_np = np.expand_dims(s_np, 0)
            source = torch.from_numpy(s_np.astype(np.float32)).permute(0, 3, 1, 2).to('cuda')
    
            # PET data
            pet_data = io.imread(join(pet_img_path, name))
            # Select the first three channels explicitly
            pet_data = pet_data[:, :, :3]
            pet_s_np = pet_data / np.max(pet_data)
            pet_s_np = (pet_s_np - 0.5) * 2.0
            assert pet_s_np.shape == (256, 256, 3), f"Shape error! Current shape: {pet_data.shape}"
            pet_s_np = np.expand_dims(pet_s_np, 0)
    
            # mask data
            mask_data = io.imread(join(mask_img_path, name))
            # Select the first three channels explicitly
            mask_data = mask_data[:, :, :3]
            mask_s_np = mask_data / np.max(mask_data)
            mask_s_np = (mask_s_np - 0.5) * 2.0
            assert mask_s_np.shape == (256, 256, 3), f"Shape error! Current shape: {mask_data.shape}"
            mask_s_np = np.expand_dims(mask_s_np, 0)
    
            # Starting the diffusion models
            ## Adding the noise to the source image CT
            noise = s_diffusion.ddim_reverse_sample_loop(
                s_model, source,
                clip_denoised=False,
                device=dist_util.dev(),
            )
    
            ## Reconstructing the CT image from noise
            source_recon = s_diffusion.ddim_sample_loop(
                s_model, (args.batch_size, 3, args.image_size, args.image_size),
                noise=noise,
                clip_denoised=False,
                device=dist_util.dev(),
            )
    
            ## Generating the pseudo PET scan from the noised CT image
            target = t_diffusion.ddim_sample_loop(
                t_model, (args.batch_size, 3, args.image_size, args.image_size),
                noise=noise,
                clip_denoised=False,
                device=dist_util.dev(),
            )
    
            # Calculate SSIM and PSNR and FID for PET vs Pseudo-PET
            target_sample = sample2img(target)
            ssim_value_pet = ssim(target_sample, pet_data, multichannel=True)
            # ssim_value_pet = ssim(target_sample, pet_data, channel_axis=True)
            print(f'ssim value: {ssim_value_pet}')
            psnr_value_pet = psnr(target_sample, pet_data)
            print(f'psnr value: {psnr_value_pet}')
            fid_value_pet = fid(target_sample, pet_data)
            print(f'fid value: {fid_value_pet}')
    
            # Calculate SSIM and PSNR and FID for CT vs Pseudo-CT
            source_sample = sample2img(source_recon)
            ssim_value_ct = ssim(source_sample, ct_data, multichannel=True)
            # ssim_value_ct = ssim(source_sample, ct_data, channel_axis=True)
            print(f'ssim value: {ssim_value_ct}')
            psnr_value_ct = psnr(source_sample, ct_data)
            print(f'psnr value: {psnr_value_ct}')
            fid_value_ct = fid(source_sample, ct_data)
            print(f'fid value: {fid_value_ct}')
    
            # %% plot
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
            images = [ct_data, color.rgb2gray(sample2img(noise)), source_sample, target_sample, mask_data,
                      pet_data]
            titles = ['CT image', 'CT noise encode',
                      'CT reconstruction', 'CT2PET', 'Mask image', 'PET image']
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i], cmap='gray')
                ax.set_title(titles[i])
                ax.axis('off')
                ax.axis('off')
            plt.suptitle(name)
            plt.savefig(join(save_path, name), dpi=300)
    
            # Record the end time
            end_time = datetime.now()
            print('ending time:', end_time)
            # Calculate time taken
            time_taken = end_time - start_time
            print('time taken:', time_taken)
            hours, remainder = divmod(time_taken.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
    
            # Write metrics and time information to the CSV file
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, ssim_value_pet, psnr_value_pet, fid_value_pet,
                                 ssim_value_ct, psnr_value_ct, fid_value_ct,
                                 start_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 end_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 f'{hours} hours, {minutes} minutes, {seconds} seconds'])
