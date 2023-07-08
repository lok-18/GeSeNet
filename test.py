import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from Datasets import datasets
from model import GeSeNet
from torch.autograd import Variable
from PIL import Image
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB
import time

def main():
    model_path = './model/GeSeNet.pth'
    model = GeSeNet(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        model.to(device)
    model.load_state_dict(torch.load(model_path))
    print('Start testing:')

    # ir -> mri   vi -> ct pet spect

    # ir_path = './test_images/MRI_CT/MRI'
    # vi_path = './test_images/MRI_CT/CT'

    # ir_path = './test_images/MRI_PET/MRI'
    # vi_path = './test_images/MRI_PET/PET'

    ir_path = './test_images/MRI_SPECT/MRI'
    vi_path = './test_images/MRI_SPECT/SPECT'

    test_dataset = datasets(ir_path=ir_path, vis_path=vi_path)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = model(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )

            fused_image = np.uint8(255.0 * fused_image)
            st = time.time()
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                ed = time.time()
                print('file_name: {0}'.format(save_path))
                print('Time:', ed - st)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()

    # fused_dir = './results/MRI_CT/'
    # fused_dir = './results/MRI_PET/'
    fused_dir = './results/MRI_SPECT/'

    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    main()
