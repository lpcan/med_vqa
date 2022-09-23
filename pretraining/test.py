from PIL import Image
import glob

files = glob.glob("/srv/scratch/z5214005/roco-dataset/data/train/radiology/images/*")
files.append(glob.glob("/srv/scratch/z5214005/roco-dataset/data/test/radiology/images/*"))
files.append(glob.glob("/srv/scratch/z5214005/roco-dataset/data/validation/radiology/images/*"))

for file in files:
    try:
        # Open the image
        img = Image.open(file)

        # Resize to 256x256
        img = img.resize((256,256))
    except OSError as err:
        print(err)
        print(f"Error with file {file}")
    
