
import os

from IPython.display import Image, display
#alternative method use os.walk

# Folder you want to scan
folder_path = "/content/data/pizza_steak_sushi/test/pizza"

# Image extensions to look for
image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

for filename in os.listdir(folder_path):
    # Get file extension
    _, ext = os.path.splitext(filename)
    if ext.lower() in image_extensions:
        print(filename)  # or use os.path.join(folder_path, filename)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")):
          img_path = os.path.join(folder_path, filename)
          display(Image(img_path))
          # img.show()   # Opens the image in the default viewer


'''

2508636.jpg
is a burger 
25 images
"/content/data/pizza_steak_sushi/test/pizza"

19 images
/content/data/pizza_steak_sushi/test/steak

31 images
/content/data/pizza_steak_sushi/test/sushi



2576168.jpg is a cake
2154394.jpg is unknownx
2785084.jpg is just fries
3821701.jpg is tea
78 images

/content/data/pizza_steak_sushi/train/pizza


75 images
/content/data/pizza_steak_sushi/train/steak


72 images
/content/data/pizza_steak_sushi/train/sushi

'''