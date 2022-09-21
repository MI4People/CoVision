from PIL import Image
import numpy as np
import os
import glob
from skimage.util import img_as_float



def get_mean_std(path, img_type=None):

    """
    Calculates mean and std of images stored in a path - default type is float (https://scikit-image.org/docs/dev/user_guide/data_types.html)

    path: path to images
    img_type: type of image - default=None means it calculates mean and std for float img
    """

    mean_sum = 0
    std_sum = 0
    n_imgs = 0

    print("path: ", path)

    print("len: ", len(glob.glob(f"{path}/*")))

    for i, image in enumerate(glob.glob(f"{path}/*")):

        with Image.open(image).convert('L') as image:
        
            if img_type == "uint8":
                data = np.asarray(image)
                print("inside")
            else:
                data = np.asarray(image)
                data = img_as_float(data)

            mean = np.mean(data)
            std = np.std(data)

            mean_sum += mean
            std_sum += std
            n_imgs = i+1

    mean = mean_sum/(n_imgs)
    std = std_sum/(n_imgs)

    print("mean: ", mean, "std: ", std)

    return mean, std

if __name__ == "__main__":

    path = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\BUS_classification\resized\train_val\256\500\images"

    #img_type="uint8"

    get_mean_std(path)