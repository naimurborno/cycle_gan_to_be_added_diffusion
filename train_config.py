# Training Config file
# 2D Cycle GAN configuration file 

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######
config['data_path'] = "/content/drive/MyDrive/Dataset/Datalast" # Path for the images
config['model_name'] = "D:/Zakaria/3T to 7T MRI/content/ResultSumon"   # choose a unique name for result folder 
config['resume_ckpt'] = None          # name of the checkpoint from which you want to resume. Otherwise None.
config['paired'] = True                     # For Aligned task set to True. Otherwise False    
config['device']='cuda'
config['batch_size']  = 1                   # batch size, Change to fit hardware
config['n_epochs']  = 1 # old system
config['n_lin_epoch'] = 100                 # number of epoch for linear learning rate
config['n_dec_epoch'] = 100                 # number of epoch for learning rate to decay
# total epoch =  n_lin_epoch + n_dec_epoch
config['save_freq'] = 5                     # epochs for saving model weights
config['gen_model'] = 'resnet_gen_9'        # ['resnet_gen_9','resnet_gen_7','resnet_gen_3']
config['dis_model'] = 'patchGAN'            # ['patchGAN','pixelGAN']

config['dis_loss'] = 'mse_loss'             # Discriminator Loss: "mse_loss" 
config['gen_loss'] = 'mse_loss'             # Generator Loss: "mse_loss"
config['identity_loss'] = 'mae_loss'        # Identity Loss: "mae_loss", "mse_loss"
config['cyc_loss'] = 'mae_loss'             # Cycle Consistency Loss: "mae_loss", "mse_loss"

config['gen_opt'] = 'adam'                  # Generator Optimizer:'adam'
config['dis_opt'] = 'adam'                  # Discriminator Optimizer:'adam'
config['gen_lr'] = 2e-4                     # Generator learning rate 
config['dis_lr'] = 2e-4                     # Generator learning rate 

### Metrics to compute
config["metrics"] = ["fid","psnr"]
# config["metrics"] = ["fid","psnr","ssim"]  # ['ssim', 'fid', 'psnr']. set it to [] for no calc