# Inference Config
# 2D Cycle GAN configuration file

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######

config["data_path"] = "/content/drive/MyDrive/Dataset/Datalast"  # Path for the input images.
# keep the image in either "Test" or "Val" subfolder
config["sub_fold"] = "Test"  # 'Test' or'Val'
config["model_name"] = "/content/cycle-gan-diffsuion/D:/Zakaria/3T to 7T MRI/content/ResultSumon"  # folder where result is saved

# List of checkpoint filenames
config["ckpt_names"] = [
    'last.ckpt'
]

config["paired"] = False  # For Aligned task set to True. Otherwise False
config["batch_size"] = 1  # batch size. keep it to 1 for least bugs
config["gen_model"] = "resnet_gen_9"  # ['resnet_gen_9','resnet_gen_7','resnet_gen_3']
config["dis_model"] = "pixelGAN"  # ['patchGAN','pixelGAN']

### Metrics to compute
# config["metrics"] = ["fid","psnr","ssim"]  # ['ssim', 'fid', 'psnr']. set it to [] for no cal
