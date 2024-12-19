import imageio.v3 as iio 
import os
import re



def create_gif(remove = False):
    image_folder = "../../Local/Plots"
    save_to_path = "./plot6.gif"
    image_folder_list = os.listdir(image_folder)
    image_list = []
    image_folder_list = sorted(image_folder_list, key=lambda x: int(re.search(r"(?<=ep_)\d+", x).group()))
    try:
        for image in image_folder_list:
            # print(image)
            
            if ".png" not in image:
                continue
            image_list.append(iio.imread(f"{image_folder}/{image}"))

            iio.imwrite(uri=save_to_path, image=image_list, duration=500, loop=0)

        print("Done")
        if remove:
            print("Function is set to remove source images when done...")
            for image in image_folder_list:
                print(f"removing... {image}")
                os.remove(f"{image_folder}/{image}")
    except: 
        print("Program failed")


create_gif()