import transforms


train_val_split = 0.8
data = "Datasets/ImageClef-2019-VQA-Med-Training/"
img_dir = "Datasets/ImageClef-2019-VQA-Med-Training/Train_images/"
val_data = "Datasets/ImageClef-2019-VQA-Med-Validation/"
val_img_dir = "Datasets/ImageClef-2019-VQA-Med-Validation/Val_images/"
wv_path = "bio_embedding_extrinsic"
batch_size = 64
epochs = 50
transform = transforms.transform