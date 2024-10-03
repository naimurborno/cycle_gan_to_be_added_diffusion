# @title utils.py

import random
import torch
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if (
                self.num_imgs < self.pool_size
            ):  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if (
                    p > 0.5
                ):  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(
                        0, self.pool_size - 1
                    )  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def set_requires_grad(nets, requires_grad):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def load_weights(ckpt, net):
    # load states
    target_state = torch.load(ckpt)
    target_state = target_state["state_dict"]
    current_state = net.state_dict()  # in case of comparison

    net.load_state_dict(target_state)
    return net


def plot_graph(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(10, 10))
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(save_path, dpi=100)


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

def convert_to_255(x:torch.Tensor):
    x = (x+1) * 127.5 # convert from [-1,1] to [0,255]
    x = x.to(torch.uint8) # convert to uint8 type
    return x


def save_paired_image(imgA, imgB, fakeB, fakeA, path):
    [imgA, imgB, fakeA, fakeB] = [
        [toImage(f) for f in X] for X in [imgA, imgB, fakeA, fakeB]
    ]

    new_image = Image.new(
        "RGB", (2 * imgA[0].size[0], 2 * imgA[0].size[0]), (250, 250, 250)
    )
    new_image.paste(imgA[0], (0, 0))
    new_image.paste(fakeB[0], (imgA[0].size[0], 0))
    new_image.paste(imgB[0], (0, imgA[0].size[0]))
    new_image.paste(fakeA[0], (imgA[0].size[0], imgA[0].size[0]))

    new_image.save(path)


def save_unpaired_image(imgA, fakeB, path):
    [imgA, fakeB] = [[toImage(f) for f in X] for X in [imgA, fakeB]]

    # new_image = Image.new(
    #     "RGB", (2 * imgA[0].size[0], imgA[0].size[0]), (250, 250, 250)
    # )
    # new_image.paste(imgA[0], (0, 0))
    # new_image.paste(fakeB[0], (imgA[0].size[0], 0))

    fakeB[0].save(path)
# def save_unpaired_image(fakeB, path):
#     # Convert fakeB images to PIL Image objects
#     fakeB = [toImage(f) for f in fakeB]

#     # Resize the first image in fakeB to 512x512 pixels
#     resized_image = fakeB[0].resize((512, 512))

#     # Save the resized image
#     resized_image.save(path)


def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def get_proper_df(df, metric, step="epoch"):
    columns = list(df["metric"].unique())
    assert metric not in ["epoch", "step"], f"{metric} cannot be 'epoch' or'step'"
    assert metric in list(columns), f"{metric} is not a valid metric"
    assert step in ["epoch", "step"], f"{step} should be either 'epoch' or'step'"
    assert metric.split("_")[-1] == step, f"{metric} is incompatible with step = {step}"

    idx = df["metric"] == metric
    temp = df[idx].copy()

    if step == "epoch":
        idx = df["metric"] == step
        epoch = df[idx].copy()
        epoch.columns = ["metric", "epoch", "step"]

        temp = (
            pd.merge(temp, epoch[["epoch", "step"]], on="step", how="left")
            .drop_duplicates()
            .reset_index(drop=True)
        )

        temp['epoch'] = temp.index

    temp = temp.drop("metric", axis=1)  # drop the metric column. it just has names

    return temp
