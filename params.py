import torchvision.transforms as transforms

train_val_split = 0.8
data = "../Datasets/VQA-RAD/"
wv_path = "bio_embedding_extrinsic"
batch_size = 64
epochs = 100
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225])])