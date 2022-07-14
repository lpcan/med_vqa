import transforms


train_val_split = None # Not necessary if using separate training and validation sets
# Various datasets
train_data = "Datasets/ImageClef-2019-VQA-Med-Training/"
train_img_dir = "Datasets/ImageClef-2019-VQA-Med-Training/Train_images/"
val_data = "Datasets/ImageClef-2019-VQA-Med-Validation/"
val_img_dir = "Datasets/ImageClef-2019-VQA-Med-Validation/Val_images/"
wv_path = "bio_embedding_extrinsic"
batch_size = 64
epochs = 50
train_transform = transforms.train_transform
val_transform = transforms.val_transform