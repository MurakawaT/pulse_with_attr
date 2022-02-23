import pandas as pd
import numpy as np
from PIL import Image
import os
from glob import glob


def get_latent_ave_dict(latent_dir, attr_path, latent_ignore_list = []):

    attr_list = pd.read_csv(attr_path, header=1, index_col=0, sep=" +", engine="python", dtype="int64")

    want_attr = attr_list.columns.tolist()
    # want_attr = ["Male","Smiling","Young"]

    latent_path_list = glob(latent_dir + "/*.npy")

    latent_dict = {}
    latent_addcount = {} # 後で消す

    latent_dict_path = "average_latent_dict.npy"

    if os.path.exists(latent_dict_path):
        print("use : ",latent_dict_path)
        latent_dict = np.load(latent_dict_path, allow_pickle='TRUE').item()
    else:
        print("make : ", latent_dict_path)
        for i, latent_path in enumerate(latent_path_list):
            latent_np_tmp = np.load(latent_path)
            related_img_name = latent_path.replace(os.sep,'/').split("/")[-1]
            if related_img_name in latent_ignore_list:
                print("ignore : ",related_img_name)
                continue
            related_img_name = related_img_name.replace("npy","jpg")
            for j, attr_name in enumerate(want_attr):

                attr_tmp = attr_list[related_img_name:related_img_name][attr_name].values.astype("int64")
                flag = 1 if attr_tmp[0] > 0 else 0

                target = attr_name + "-" + str(flag)
                count = latent_addcount.get(target,0)

                if count == 0:
                    latent_dict[target] = latent_np_tmp
                else:
                    latent_dict[target] = (count * latent_dict[target] + latent_np_tmp) / (count+1)
                
                latent_addcount[target] = count+1
                latent_dict[target+"_addcount"] = count+1
        np.save(latent_dict_path,latent_dict)

    # return [ attr1_positive, attr1_negative, attr2_positive, ...]
    return (latent_dict,latent_addcount)
