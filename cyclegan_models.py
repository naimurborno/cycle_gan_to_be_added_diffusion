import torch
import gc
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import os

from generator import get_generator
from discriminator import get_model
from utils import ImagePool, init_weights, set_requires_grad, load_weights, convert_to_255
from metrics import get_metric_class, calc_metric
#@title CycleGan.py
class CycleGan(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False #this has been changed due to automatic optimizer problem.
        self.save_hyperparameters()
        self.config = config

        #stable diffusion
        self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        self.sd_pipeline.safety_checker=None
        # self.sd_pipeline.set_logging_level(logging.ERROR) 
        self.sd_pipeline.to(self.config['device'])  # Ensure the model is on the correct device
        self.sd_pipeline.set_progress_bar_config(leave=False)
        self.sd_pipeline.set_progress_bar_config(disable=True)




        # Set metrics
        metric_list = config['metrics']        
        self.metric_list = [metric.lower() for metric in metric_list] # Lower it just incase
        # Set Metric Dict
        self.metric_dict = {}
        for metric in self.metric_list:
            self.metric_dict[f'{metric}_A'] = get_metric_class(metric).to('cuda')
            self.metric_dict[f'{metric}_B'] = get_metric_class(metric).to('cuda')
        self.metric_dict = torch.nn.ModuleDict(self.metric_dict)
        # generator pair
        self.genX = get_generator(config['gen_model'])
        self.genY = get_generator(config['gen_model'])
        
        # discriminator pair
        self.disX = get_model(config['dis_model'])
        self.disY = get_model(config['dis_model'])
        
        self.lm = 10.0
        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()
        self.genLoss = None
        self.disLoss = None

        for m in [self.genX, self.genY, self.disX, self.disY]:
            init_weights(m)
    
    def configure_optimizers(self):
        optG = Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=self.config['gen_lr'], betas=(0.5, 0.999))
        
        optD = Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=self.config['dis_lr'], betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - self.config['n_lin_epoch']) / (self.config['n_dec_epoch']+1)
        #gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / (100+1)
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.mse_loss(predictions, target)
    
    def get_mae_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.l1_loss(predictions, target)
    def refine_with_stable_diffusion(self, image_tensor):
        # Convert the tensor to a PIL image for Stable Diffusion
        pil_image = self.toImage(image_tensor)
        # Use Stable Diffusion to refine the image
        with torch.no_grad():
            refined_image = self.sd_pipeline(
                prompt="",
                image=pil_image,  # This is the input image
                num_inference_steps=10,
                strength=0.75,  # Control the strength of diffusion, higher is more transformed
                guidance_scale=7.5  # Control adherence to the original image
            ).images[0]
        # Convert the refined image back to a tensor
        refined_tensor = self.image_to_tensor(refined_image).to(self.config['device'])
        return refined_tensor        
    def generator_training_step(self, imgA, imgB):        
        """cycle images - using only generator nets"""
        fakeB = self.genX(imgA)
        cycledA = self.genY(fakeB)
        
        fakeA = self.genY(imgB)
        cycledB = self.genX(fakeA)
        
        sameB = self.genX(imgB)
        sameA = self.genY(imgA)
        
        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')
        
        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, 'real')
        
        #stable diffusion post-processing
        fakeB = self.refine_with_stable_diffusion(fakeB)
        fakeA = self.refine_with_stable_diffusion(fakeA)
        
        # compute extra losses
        if self.config['identity_loss'] == "mae_loss": 
            identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)
        elif self.config['identity_loss'] == "mse_loss": 
            identityLoss = F.mse_loss(sameA, imgA) + F.mse_loss(sameB, imgB)
        else:
            raise NotImplementedError(f'This Identity Loss is not implemented')
        
        # compute cycleLosses
        if self.config['cyc_loss'] == "mae_loss": 
            cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)
        elif self.config['cyc_loss'] == "mse_loss": 
            cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)
        else:
            raise NotImplementedError(f'This Cycle Consistency Loss is not implemented')
        # cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)
        
        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        self.genLoss = mseGenA + mseGenB + self.lm * extraLoss
        # Progress Bar Logging
        self.log('_gen_loss', self.genLoss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('_id_loss', identityLoss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('_cyc_loss', cycleLoss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # CSV Logging
        self.log('gen_loss', self.genLoss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('id_loss', identityLoss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('cyc_loss', cycleLoss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('gen_A_loss', mseGenA.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('gen_B_loss', mseGenB.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # Compute and log Metrics
        if self.metric_list: # only compute if list is not empty
            self.compute_metrics(imgA, imgB, fakeB, fakeA)
        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        
        return self.genLoss
    
    def discriminator_training_step(self, imgA, imgB):
        """Update Discriminator"""        
        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)
        
        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, 'real')
        
        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')
        
        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')
        
        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeB, 'fake')
        
        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)
        # Progress Bar Logging
        self.log('_dis_loss', self.disLoss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # CSV Logging
        self.log('dis_loss', self.disLoss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mse_real_A', mseRealA.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mse_real_B', mseRealB.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mse_fake_A', mseFakeA.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mse_fake_B', mseFakeB.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return self.disLoss
    
    def training_step(self, batch, batch_idx):  
      imgA, imgB = batch['A'], batch['B']

      # Get the optimizers
      optG, optD = self.optimizers()

      # Generator step
      set_requires_grad([self.disX, self.disY], False)  # Freeze discriminators
      gen_loss = self.generator_training_step(imgA, imgB)
      optG.zero_grad()
      self.manual_backward(gen_loss)
      optG.step()

      # Discriminator step
      set_requires_grad([self.disX, self.disY], True)  # Unfreeze discriminators
      dis_loss = self.discriminator_training_step(imgA, imgB)
      optD.zero_grad()
      self.manual_backward(dis_loss)
      optD.step()

      # Log losses
      # self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      # self.log('dis_loss', dis_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

      return {"loss": gen_loss + dis_loss}

        
    def save_image(self, batch, batch_idx):
        imgA, imgB = batch['A'], batch['B']
        with torch.no_grad():
            fakeB = self.genX(imgA)
            fakeA = self.genY(imgB)

        [imgA, imgB, fakeA, fakeB] = [[self.toImage(f) for f in X] for X in [imgA, imgB, fakeA, fakeB]]
        for i in range(len(imgA)):
            new_image = Image.new('RGB',(2*imgA[i].size[0], 2* imgA[i].size[0]), (250,250,250))
            new_image.paste(imgA[i],(0,0))
            new_image.paste(fakeB[i],(imgA[i].size[0],0))
            new_image.paste(imgB[i],(0,imgA[i].size[0]))
            new_image.paste(fakeA[i],(imgA[i].size[0],imgA[0].size[0]))
            # Save image
            image_dir = os.path.join(self.config['model_name'],'Train Images')
            os.makedirs(image_dir, exist_ok= True)
            image_dir = os.path.join(image_dir, f'epoch_{self.current_epoch: 05d}')
            os.makedirs(image_dir, exist_ok = True)
            new_image.save(os.path.join(image_dir,f'{batch_idx:05d}_{i}.png'))

    def toImage(self, x):#(this function has been changed)
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
    def image_to_tensor(self, image):
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

    def compute_metrics(self, imgA: torch.Tensor, imgB: torch.Tensor, fakeB: torch.Tensor, fakeA: torch.Tensor):
        # Convert to [0, 255] for calculation
        with torch.no_grad():
            imgA_255 = convert_to_255(imgA)
            imgB_255 = convert_to_255(imgB)
            fakeB_255 = convert_to_255(fakeB)
            fakeA_255 = convert_to_255(fakeA)
            # Compute Metrics
            for metric in self.metric_list:
                self.metric_dict[f'{metric}_A'] = calc_metric(self.metric_dict[f'{metric}_A'], metric, fakeA_255, imgA_255)
                self.metric_dict[f'{metric}_B'] = calc_metric(self.metric_dict[f'{metric}_B'], metric, fakeB_255, imgB_255)
                # Log them
                self.log(f'{metric}_A', self.metric_dict[f'{metric}_A'], on_step=True, on_epoch=True, prog_bar=(metric == 'ssim'), logger=True)
                self.log(f'{metric}_B', self.metric_dict[f'{metric}_B'], on_step=True, on_epoch=True, prog_bar=(metric == 'ssim'), logger=True)

    
    def on_train_epoch_end(self):
        print('\n')
        # Reset Metrics
        for metric in self.metric_list:
            self.metric_dict[f'{metric}_A'].reset()
            self.metric_dict[f'{metric}_B'].reset()
        
        gc.collect()
        # torch.cuda.empty_cache()