import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import losses
import models_arbitrary
from datareader import DBreader_Adobe240fps
from TestModule import adobe240fps
from trainer import Trainer

parser = argparse.ArgumentParser(description="AdaCoF-Pytorch")

# parameters
# Model Selection
parser.add_argument("--model", type=str, default="adacofnet")

# Hardware Setting
parser.add_argument("--gpu_id", type=int, default=0)

# Directory Setting
parser.add_argument(
    "--train", type=str, default="../DeepVideoDeblurring_Dataset/quantitative_datasets"
)
parser.add_argument("--out_dir", type=str, default="./output_adacof_train")
parser.add_argument("--load", type=str, default=None)
parser.add_argument(
    "--test_input",
    type=str,
    default="../DeepVideoDeblurring_Dataset/quantitative_datasets",
)
# parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

# Learning Options
parser.add_argument("--epochs", type=int, default=50, help="Max Epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument(
    "--loss", type=str, default="1*VGG", help="loss function configuration"
)
parser.add_argument("--patch_size_h", type=int, default=256, help="Patch size")
parser.add_argument("--patch_size_w", type=int, default=256, help="Patch size")

# Optimization specifications
parser.add_argument("--lr", type=float, default=6e-4, help="learning rate")
# parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
# parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
# parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument(
    "--optimizer",
    default="ADAM",
    choices=("SGD", "ADAM", "RMSprop", "ADAMax"),
    help="optimizer to use (SGD | ADAM | RMSprop | ADAMax)",
)
# parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for AdaCoF
# parser.add_argument('--kernel_size', type=int, default=5)
# parser.add_argument('--dilation', type=int, default=1)

# Options for network
parser.add_argument("--pooling_with_mask", type=int, default=1)
parser.add_argument("--decoder_with_mask", type=int, default=1)
parser.add_argument("--softargmax_with_mask", type=int, default=0)
parser.add_argument("--decoder_with_gated_conv", type=int, default=1)
parser.add_argument("--residual_detail_transfer", type=int, default=0)
parser.add_argument("--beta_learnable", type=int, default=0)
parser.add_argument("--splatting_type", type=str, default="softmax")
# parser.add_argument('--residual_detail_transfer_with_mask', type=int, default=0)
# parser.add_argument('--mask_with_proxy_mask', type=int, default=0)
# parser.add_argument('--max_proxy', type=int, default=0)
parser.add_argument("--concat_proxy", type=int, default=0)
parser.add_argument("--center_residual_detail_transfer", type=int, default=1)
parser.add_argument("--pooling_with_center_bias", type=int, default=1)
parser.add_argument("--pooling_type", type=str, default="gaussian")
parser.add_argument("--no_pooling", type=int, default=0)
parser.add_argument("--single_decoder", type=int, default=0)
parser.add_argument("--noDL_CNNAggregation", type=int, default=0)
parser.add_argument("--gumbel", type=int, default=0)
parser.add_argument("--inference_with_frame_selection", type=int, default=1)
parser.add_argument("--FOV_expansion", type=int, default=0)
parser.add_argument("--all_backward", type=int, default=0)
parser.add_argument("--seamless", type=int, default=0)
parser.add_argument("--bundle_forward_flow", type=int, default=0)


transform = transforms.Compose([transforms.ToTensor()])


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    dataset = DBreader_Adobe240fps(
        args.train, random_crop=[args.patch_size_h, args.patch_size_w]
    )
    TestDB = adobe240fps(args.test_input, args)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=True,
    )
    model = models_arbitrary.Model(args)
    loss = losses.Loss(args)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]

    my_trainer = Trainer(args, train_loader, TestDB, model, loss, start_epoch)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    with open(args.out_dir + "/config.txt", "a") as f:
        f.write(now + "\n\n")
        for arg in vars(args):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
        f.write("\n")

    counter = 0
    while not my_trainer.terminate():
        my_trainer.train()
        if counter > 45:
            my_trainer.test()
        counter += 1

    my_trainer.close()


if __name__ == "__main__":
    main()
