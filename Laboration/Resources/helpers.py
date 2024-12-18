import imageio.v3 as iio 
import os
import re



def create_gif():
    image_folder = "../../Local/Plots"
    save_to_path = "./plot.gif"
    image_folder_list = os.listdir(image_folder)
    image_list = []
    image_folder_list = sorted(image_folder_list, key=lambda x: int(re.search(r"(?<=ep_)\d+", x).group()))

    for image in image_folder_list:
        # print(image)
        
        if ".png" not in image:
            continue
        image_list.append(iio.imread(f"{image_folder}/{image}"))

        iio.imwrite(uri=save_to_path, image=image_list, duration=500, loop=0)

    print("Done")

create_gif()