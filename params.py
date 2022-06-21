import torchvision.transforms as transforms

train_val_split = 0.8
data = "Datasets/ImageClef-2019-VQA-Med-Training/"
img_dir = "Datasets/ImageClef-2019-VQA-Med-Training/Train_images/"
wv_path = "bio_embedding_extrinsic"
batch_size = 64
epochs = 50
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225]),
                                transforms.RandomRotation(10)])
