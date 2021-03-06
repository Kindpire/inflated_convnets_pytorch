import argparse
import copy
import json

from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.i3vgg import I3vgg


# To profile uncomment @profile and run `kernprof -lv inflate_vgg.py`
# @profile
def run_inflater(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder('data/dummy-dataset',
                                   transforms.Compose([
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    class_idx = json.load(open('data/imagenet_class_index.json'))
    imagenet_classes = [class_idx[str(k)][1] for k in range(len(class_idx))]

    if args.vgg_nb == 16:
        vgg = torchvision.models.vgg16_bn(pretrained=True)
    else:
        raise ValueError('vgg_nb should be in [16] but got {}'
                         ).format(args.vgg_nb)
    #vgg_feature_module = list(vgg.features)
    #vgg_class_module = list(vgg.classifier)
    #vgg_feature_net = torch.nn.Sequential(*vgg_feature_module)
    #vgg_class_net = torch.nn.Sequential(*vgg_class_module)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    i3vgg = I3vgg(copy.deepcopy(vgg), args.frame_nb)
    i3vgg.eval()
    i3vgg.cuda()
    vgg.eval()
    vgg.cuda()
    #vgg_feature_net.eval()
    #vgg_feature_net.cuda()
    #vgg_class_net.eval()
    #vgg_class_net.cuda()

    for i, (input_2d, target) in enumerate(loader):
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        input_2d_var = torch.autograd.Variable(input_2d.cuda())

        #out2d = vgg_feature_net(input_2d_var)
        #out2d = out2d.view(out2d.size(0), -1)
        #out2d = vgg_class_net(out2d)
        out2d = vgg(input_2d_var)
        out2d = out2d.cpu().data

        input_3d = input_2d.unsqueeze(2).repeat(1, 1, args.frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d.cuda())

        out3d = i3vgg(input_3d_var)
        out3d = out3d.cpu().data
        print(out2d.shape, out3d.shape)
        out_diff = out2d - out3d
        print('mean abs error {}'.format(out_diff.abs().mean()))
        print('mean abs val {}'.format(out2d.abs().mean()))

        # Computing errors between final predictions of inflated and uninflated
        # dense networks
        print(
            'Batch {i} maximum error between 2d and inflated predictions: {err}'.
            format(i=i, err=out_diff.max()))
        assert (out_diff.max() < 0.0001)

        if args.display_samples:
            max_vals, max_indexes = out3d.max(1)
            for sample_idx in range(out3d.shape[0]):
                sample_out = out3d[sample_idx]

                top_val, top_idx = torch.sort(sample_out, 0, descending=True)

                print('Top {} classes and associated scores: '.format(
                    args.top_k))
                for i in range(args.top_k):
                    print('[{}]: {}'.format(imagenet_classes[top_idx[i]],
                                            top_val[i]))

                sample_img = input_2d[sample_idx].numpy().transpose(1, 2, 0)
                sample_img = (sample_img - sample_img.min()) * (1 / (
                    sample_img.max() - sample_img.min()))
                plt.imshow(sample_img)
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inflates vgg and runs\
    it on dummy dataset to compare outputs from original and inflated networks\
    (should be the same)')
    parser.add_argument(
        '--vgg_nb',
        type=int,
        default=16,
        help='What version of vgg to use, in [16]')
    parser.add_argument(
        '--display_samples',
        action='store_true',
        help='Whether to display samples and associated\
        scores for 3d inflated vgg')
    parser.add_argument(
        '--top_k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')
    parser.add_argument(
        '--frame_nb',
        type=int,
        default='4',# what if less than 8? actually for vgg16 bigger than 4 will explod the CUDA memory of 1080ti
        help='Number of video_frames to use (should be a multiple of 8)')
    args = parser.parse_args()
    run_inflater(args)
