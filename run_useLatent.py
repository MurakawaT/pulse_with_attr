from PULSE_useLatent import PULSE
from dataloader import get_latent_ave_dict
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
import numpy as np



class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png")) + list(self.root_path.glob("*.jpg"))
        # self.image_list = list(self.root_path.glob("*.jpg"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if(self.duplicates == 1):
            return image,img_path.stem
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}"

parser = argparse.ArgumentParser(description='PULSE')

#I/O arguments
parser.add_argument('-input_dir', type=str, default='input', help='input data directory')
parser.add_argument('-output_dir', type=str, default='runs', help='output data directory')
parser.add_argument('-latent_dir', type=str, default='latent', help='input latent directory')
parser.add_argument('-latent_ignore_list', type=str, default='0,100', help='ignore list in input latent directory : 0.npy ~ 99.npy')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-duplicates', type=int, default=1, help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')
parser.add_argument('-attr_path', type=str, default='CelebAMask-HQ-attribute-anno.txt', help='annotation text (CelebAMask-HQ dataset)')

#PULSE with Attr arguments
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS", help='Loss function to use')
parser.add_argument('-loss_attr_str', type=str, default="1*Male+1*Smiling", help='Loss attr')
parser.add_argument('-eps', type=float, default=1e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')

kwargs = vars(parser.parse_args())

dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)
latent_dir = kwargs["latent_dir"]
attr_path = kwargs["attr_path"]

save_attrPng_name = kwargs["loss_attr_str"].replace("*","").replace("+","_")
ignore_start, ignore_end = kwargs["latent_ignore_list"].split(",")
latent_ignore_list = [str(i)+".npy" for i in range(int(ignore_start), int(ignore_end))]
latent_ave_dict, latent_add_count = get_latent_ave_dict(latent_dir, attr_path, latent_ignore_list)
# for k in latent_ave_dict.keys():
#     tmp_attrAve = latent_ave_dict[k]
#     tmp_attrAve = tmp_attrAve.mean(axis=1)
#     tmp_attrAve = tmp_attrAve.reshape(1,1,512)
#     tmp_attrAve = np.tile(tmp_attrAve,(1,18,1))
#     latent_ave_dict[k] = tmp_attrAve

dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])

model = PULSE(cache_dir=kwargs["cache_dir"])
model = DataParallel(model)

toPIL = torchvision.transforms.ToPILImage()

print(len(dataloader))

for ref_im, ref_im_name in dataloader:
    if(kwargs["save_intermediate"]):
        padding = 1+ceil(log10(kwargs["steps"]))
        for i in range(kwargs["batch_size"]):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j,(HR,LR,latent_np) in enumerate(model(ref_im, latent_ave_dict, **kwargs)):
            if j % (int(kwargs["steps"])//50) == 0:
                for i in range(kwargs["batch_size"]):
                    toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                        int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                    # toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                    #     int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")
    else:
        #out_im = model(ref_im, latent_ave_dict, **kwargs)
        for j,(HR,LR,latent_np) in enumerate(model(ref_im, latent_ave_dict, **kwargs)):
            print(ref_im_name)
            for i in range(kwargs["batch_size"]):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                    out_path / f"{ref_im_name[i]}_{save_attrPng_name}.png")

                # toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                #     out_path / f"{ref_im_name[i]}_{save_attrPng_name}_low.png")
                
