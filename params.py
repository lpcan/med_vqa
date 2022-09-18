import transforms

# Parameters
k_folds = 5 # Set to 0 if no cross-validation is to be performed. If performed, all data must be in train_data directory
train_val_split = 0.8 # Not necessary if using separate training and validation sets
batch_size = 64
epochs = 50
train_transform = transforms.train_transform
val_transform = transforms.val_transform
num_attn_layers = 1
lr = 1e-5

# Various datasets
train_data = "Datasets/2019-VQA-Med-All/"
train_img_dir = "Datasets/ImageClef-2019-VQA-Med-Training/Train_images/"
val_data = "Datasets/ImageClef-2019-VQA-Med-Validation/"
val_img_dir = "Datasets/ImageClef-2019-VQA-Med-Validation/Val_images/"
wv_path = "bio_embedding_extrinsic"