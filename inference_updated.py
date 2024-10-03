# Imports
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
# Pipeline Imports
import infer_config_updated
from diffusers import StableDiffusionImg2ImgPipeline
from UnpairedDataset import UnpairedDataset
from PairedDataset import PairedDataset
from cyclegan_models import CycleGan
from utils import save_paired_image, save_unpaired_image, load_weights, convert_to_255

import warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
def toImage(x):#(this function has been changed)
        if x.ndim == 4:
          x = x.squeeze(0)  # Remove batch dimension if present
    # Ensure x has 3 dimensions
        if x.ndim == 3:
            x = x.clone().detach().cpu().numpy()
            x = np.transpose(x, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            x = (x + 1) * 127.5
            x = x.astype('uint8')
            return Image.fromarray(x)
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
def image_to_tensor(image):
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
def refine_with_stable_diffusion(image_tensor):
        # Convert the tensor to a PIL image for Stable Diffusion
        pil_image =toImage(image_tensor)
        # Use Stable Diffusion to refine the image
        with torch.no_grad():
            refined_image =sd_pipeline(
                prompt="",
                image=pil_image,  # This is the input image
                num_inference_steps=10,
                strength=0.75,  # Control the strength of diffusion, higher is more transformed
                guidance_scale=7.5  # Control adherence to the original image
            ).images[0]
        # Convert the refined image back to a tensor
        refined_tensor = image_to_tensor(refined_image).to('cuda')
        return refined_tensor
  

if __name__ == "__main__":
    # Get the config file
    config = infer_config_updated.config 
    root = config['data_path']
    mode = config['sub_fold']
    sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    sd_pipeline.safety_checker=None
        # self.sd_pipeline.set_logging_level(logging.ERROR) 
    sd_pipeline.to('cuda')  # Ensure the model is on the correct device
    sd_pipeline.set_progress_bar_config(leave=False)
    sd_pipeline.set_progress_bar_config(disable=True)


    # Set paired or unpaired
    if config['paired']:
        test_ds = PairedDataset(root, mode)
    else:
        test_ds = UnpairedDataset(root, mode)

    # Create Dataloader
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    # Iterate through checkpoints
    for ckpt_name in config["ckpt_names"]:
        # Model Loaded
        print(f"Loading model from -> {os.path.join(config['model_name'], ckpt_name)}\n")
        model = CycleGan.load_from_checkpoint(os.path.join(config['model_name'], ckpt_name))

        # Set model to 'cuda' and evaluation mode
        model.eval()
        model.to('cuda')

        # Extract epoch number for folder naming
        epoch_number =1# ckpt_name.split('=')[1].split('-')[0]

        # Iterate through batches
        for i, batch in enumerate(tqdm(test_dl, total=len(test_dl))):
            imgA, imgB = batch['A'].to('cuda'), batch['B'].to('cuda')        
            with torch.no_grad():
                fakeB = model.genX(imgA)
                # fakeB=refine_with_stable_diffusion(fakeB)
                # fakeB=fakeB.unsqueeze(0) 
                fakeA = model.genY(imgB) 
                # fakeA=refine_with_stable_diffusion(fakeA)
                # fakeA=fakeA.unsqueeze(0) 
                # print(fakeA.shape)

            # Convert back to cpu for saving image
            imgA, imgB, fakeB, fakeA = imgA.cpu(), imgB.cpu(), fakeB.cpu(), fakeA.cpu()
            pathA, pathB = batch['pathA'], batch['pathB']
            for i in range(len(pathA)):
                temp_pathA = os.path.split(pathA[i])[-1]
                temp_pathB = os.path.split(pathB[i])[-1]
                if config['paired']:
                    path_dir = os.path.join(config['model_name'], f'Epoch_{epoch_number}_{config["sub_fold"]}_Predictions')
                    os.makedirs(path_dir, exist_ok=True)
                    path = os.path.join(path_dir, temp_pathA)
                    save_paired_image(imgA, imgB, fakeB, fakeA, path)
                else:
                    # For A
                    path_dir = os.path.join(config['model_name'], f'Epoch_{epoch_number}_{config["sub_fold"]}_Predictions_A')
                    os.makedirs(path_dir, exist_ok=True)
                    path = os.path.join(path_dir, temp_pathA)
                    save_unpaired_image(imgA, fakeB, path)
                    # For B 
                    path_dir = os.path.join(config['model_name'], f'Epoch_{epoch_number}_{config["sub_fold"]}_Predictions_B')
                    os.makedirs(path_dir, exist_ok=True)
                    path = os.path.join(path_dir, temp_pathB)
                    save_unpaired_image(imgB, fakeA, path)

        print(f"Inference Completed for epoch {epoch_number}!")
