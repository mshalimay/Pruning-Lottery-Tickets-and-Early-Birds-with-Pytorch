import models
import utils as u

checkpoint_path = "./checkpoints/resnetb_cifar10_depth=20_batch=64_epoch=160"
args.dataset = "cifar10"
# load and preprocess data
data_train, data_test = u.load_preprocess_data(args.dataset, args)

# create data loaders
train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
test_loader = torch.utils.data.DataLoader(data_test, **test_kwargs)

# Load the model
model = resnet(depth=20, num_classes=10)