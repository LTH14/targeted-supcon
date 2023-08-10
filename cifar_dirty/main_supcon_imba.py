from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import datasets
import torch.nn.functional as F

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import align_loss, uniform_loss
from networks.resnet_big import SupConResNet
from networks.resnet_small import SupConResNet_s
from losses import SupConLoss
import json
import numpy as np
import random

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

OPTIMAL_10_3 = [[0.0965204948, -0.9922570548, 0.0781647696],
                [-0.8228686347, 0.5681892222, 0.0069439062],
                [-0.0047906412, 0.7093750043, -0.7048149779],
                [-0.3024879498, -0.1351572898, 0.9435218849],
                [0.0810324929, 0.8180949724, 0.5693455464],
                [0.8592024490, 0.5016328955, -0.1006756660],
                [-0.4238601901, -0.2889105431, -0.8584132090],
                [0.7311387811, -0.2853071717, 0.6197063019],
                [-0.8592024492, -0.5016328953, 0.1006756661],
                [0.6453156470, -0.3940271396, -0.6544542222]]


OPTIMAL_10_10 = np.load('optimal_10_10.npy').tolist()
OPTIMAL_10_128 = np.load('optimal_10_128.npy').tolist()
OPTIMAL_10_128_yy = np.load('optimal_10_128_yy.npy').tolist()
OPTIMAL_100_128 = np.load('optimal_100_128.npy').tolist()
OPTIMAL_100_128_yy = np.load('optimal_100_128_yy.npy').tolist()


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--class_ratio', type=str, default='', help='class list for training')
    parser.add_argument('--imb_rate', type=float, default=0.1, help='maximum imbalance rate')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--feat_dim', type=int, default=128, help='feature dim for mlp head')
    parser.add_argument('--K', type=int, default=0, help='KCL positive number')
    parser.add_argument('--target_repeat', type=int, default=25, help='target repeat time')
    parser.add_argument('--no_simclr', action='store_true',
                        help='no augmentation simclr')
    parser.add_argument('--use_target', action='store_true',
                        help='use target for supcon')
    parser.add_argument('--only_t', action='store_true',
                        help='only use target')
    parser.add_argument('--name', type=str, default='', help='name for model')
    parser.add_argument('--balanced_sample', action='store_true',
                        help='balanced sampling input from classes')
    parser.add_argument('--weighted', action='store_true',
                        help='using weighted supcon')
    parser.add_argument('--unbiased', action='store_true',
                        help='using unbiased supcon')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--save_feature', action='store_true',
                        help='save features')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training.')

    opt = parser.parse_args()
    return opt


def parse_option_stage1(opt):
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '/data/netmit/rf-diary2/{}/datasets'.format(opt.dataset)
    opt.model_path = '/data/netmit/SenseFS/targeted/{}/model'.format(opt.dataset)
    opt.tb_path = '/data/netmit/SenseFS/targeted/{}/model'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    class_ratio = opt.class_ratio.split(',')
    opt.class_ratio = list([])
    if len(class_ratio) == 1:
        opt.class_ratio = []
    else:
        for one_ratio in class_ratio:
            opt.class_ratio.append(int(one_ratio))

    opt.model_name = '{}_{}_lr_{}_wd_{}_bsz_{}_temp_{}_seed_{}_ir_{}_fd_{}_tgted_{}_only_{}_tr_{}_' \
                     'bala_{}_weighted_{}_unbiased_{}_k_{}_ep_{}_{}'. \
        format(opt.method, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.seed, opt.imb_rate, opt.feat_dim,
               opt.use_target, opt.only_t, opt.target_repeat, opt.balanced_sample, opt.weighted, opt.unbiased,
               opt.K, opt.epochs, opt.name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.save_feature:
        opt.model_name = '{}_savefeature'.format(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    print("Saving to {}".format(opt.save_folder))

    if not opt.save_feature:
        # save code
        code_path = os.path.join(opt.save_folder, 'code')
        os.makedirs(code_path)
        cmd = "cp -r . {}".format(code_path)
        os.system(cmd)

        # save args
        all_opt = vars(opt)
        f_log = open(os.path.join(opt.save_folder, 'args.txt'), 'w')
        for key in sorted(all_opt.keys()):
            f_log.write("{:15} {}\n".format(key, all_opt[key]))
        f_log.close()

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    if opt.n_cls == 10:
        opt.target_repeat = opt.target_repeat*np.ones(opt.n_cls, dtype=np.int32)
    elif opt.n_cls == 100:
        opt.target_repeat = opt.target_repeat*np.ones(opt.n_cls, dtype=np.int32)
    else:
        raise NotImplementedError
    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        if opt.imb_rate == 1:
            print('Use CIFAR100 Normalization')
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        else:
            print('Use CIFAR10 Normalization')
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder, class_ratio=opt.class_ratio,
                                         imb_rate=opt.imb_rate,
                                         balanced_sample=opt.balanced_sample,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder, class_ratio=opt.class_ratio,
                                       imb_rate=opt.imb_rate,
                                       balanced_sample=opt.balanced_sample,
                                       train=False,
                                       transform=TwoCropTransform(val_transform))
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder, class_ratio=opt.class_ratio,
                                          imb_rate=opt.imb_rate,
                                          balanced_sample=opt.balanced_sample,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder, class_ratio=opt.class_ratio,
                                        imb_rate=opt.imb_rate,
                                        balanced_sample=opt.balanced_sample,
                                        train=False,
                                        transform=TwoCropTransform(val_transform))
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                             transform=TwoCropTransform(train_transform))
        val_dataset = datasets.ImageFolder(root=opt.data_folder,
                                           transform=TwoCropTransform(val_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    if opt.model == 'resnet50':
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim)
    elif opt.model == 'resnet32':
        model = SupConResNet_s(name=opt.model, feat_dim=opt.feat_dim)
    else:
        raise NotImplementedError
    criterion = SupConLoss(temperature=opt.temp, unbiased=opt.unbiased)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if opt.ckpt is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

    else:
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        if 'model' in ckpt.keys():
            state_dict = ckpt['model']
        else:
            state_dict = ckpt['state_dict']
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            else:
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

            if 'model' in ckpt.keys():
                model.load_state_dict(state_dict)
            else:
                model.encoder.load_state_dict(state_dict, strict=False)

    return model, criterion


def loop(loader, model, criterion, optimizer, epoch, opt, train=True, train_data=True, save_feature=False):
    """one epoch training"""
    if train:
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    alignment_meter = AverageMeter()
    uniformity_meter = AverageMeter()

    end = time.time()
    feature_dict = {}

    if opt.weighted:
        if not len(opt.class_ratio) == opt.n_cls:
            cls_num = opt.n_cls
            class_ratio_exp = []
            for cls_idx in range(cls_num):
                ratio = opt.imb_rate ** (cls_idx / (cls_num - 1.0))
                class_ratio_exp.append(ratio)

            cls_num_list = []
            img_max = 50000 / opt.n_cls
            for ratio in class_ratio_exp:
                cls_num_list.append(int(img_max * ratio))

            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            weight = per_cls_weights
            # weight = 1 / np.array(class_ratio_exp) / 5
        else:
            weight = 1 / np.array(opt.class_ratio)
    else:
        weight = np.ones(opt.n_cls)

    for idx, (images, labels) in enumerate(loader):
        data_time.update(time.time() - end)

        if opt.no_simclr:
            images = images[0]
        else:
            images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        if train:
            warmup_learning_rate(opt, epoch, idx, len(loader), optimizer)

        # compute loss
        features_contrast, features = model(images)
        features_contrast_original = features_contrast.clone()
        if opt.no_simclr:
            f1 = features_contrast
            f2 = features_contrast
            features_contrast = features_contrast.unsqueeze(1)
            target_mask = None
            target_index = None
        else:
            f1, f2 = torch.split(features_contrast, [bsz, bsz], dim=0)
            if opt.use_target:
                assert len(opt.target_repeat) == opt.n_cls
                # optimal_target = [[np.cos(2*i*np.pi/10), np.sin(2*i*np.pi/10)] for i in range(10)]
                # optimal_target = np.eye(10)
                if opt.n_cls == 10:
                    # optimal_target = OPTIMAL_10_3
                    optimal_target = OPTIMAL_10_128
                # optimal_target = OPTIMAL_10_10
                elif opt.n_cls == 100:
                    # optimal_target = OPTIMAL_100_128
                    optimal_target = OPTIMAL_100_128_yy
                else:
                    raise NotImplementedError

                target1 = torch.Tensor(optimal_target).cuda(non_blocking=True).float()
                target1 = torch.cat([target1[i:i+1, :].repeat(opt.target_repeat[i], 1) for i in range(len(opt.target_repeat))], dim=0)
                target2 = target1.clone()

                if opt.n_cls == 10:
                    label_order = [4, 8, 3, 1, 9, 5, 2, 7, 0, 6]
                elif opt.n_cls == 100:
                    label_order = list(range(100))
                else:
                    raise NotImplementedError
                target_labels = torch.cat([torch.Tensor([label_order[i]]).repeat(opt.target_repeat[i]) for i in range(len(opt.target_repeat))], dim=0).cuda(non_blocking=True).long()

                f1_appended = torch.cat([f1, target1], dim=0)
                f2_appended = torch.cat([f2, target2], dim=0)
                labels = torch.cat([labels, target_labels], dim=0)
                features_contrast = torch.cat([f1_appended.unsqueeze(1), f2_appended.unsqueeze(1)], dim=1)

                if opt.only_t:
                    target_mask = torch.ones_like(labels).squeeze()
                    target_mask[:-target_labels.size(0)] = 0
                else:
                    target_mask = None
                # target_index = torch.ones_like(labels).squeeze()
                # target_index[:-target_labels.size(0)] = 0
                target_index = None
            else:
                features_contrast = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                target_mask = None
                target_index = None
        if opt.method == 'SupCon':
            loss = criterion(features_contrast, labels, k=opt.K, weight=weight, target_mask=target_mask, target_index=target_index)
        elif opt.method == 'SimCLR':
            loss = criterion(features_contrast)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # alignment and uniformity loss
        alignment = align_loss(f1, f2)
        uniformity = uniform_loss(F.normalize(features[:bsz], dim=1))
        alignment_meter.update(alignment.item())
        uniformity_meter.update(uniformity.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # save feature
        for i in range(bsz):
            if int(labels[i]) not in feature_dict.keys():
                feature_dict[int(labels[i])] = [features_contrast_original[i].detach().cpu().numpy().tolist()]
            else:
                feature_dict[int(labels[i])].append(features_contrast_original[i].detach().cpu().numpy().tolist())

        # print info
        if (idx + 1) % opt.print_freq == 0:
            prefix = 'Train' if train else 'Test'
            print('{prefix}: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'alignment {alignment.val:.3f} ({alignment.avg:.3f})\t'
                  'uniformity {uniformity.val:.3f} ({uniformity.avg:.3f})'.format(
                epoch, idx + 1, len(loader), prefix=prefix, batch_time=batch_time,
                data_time=data_time, loss=losses, alignment=alignment_meter, uniformity=uniformity_meter))
            sys.stdout.flush()

    if save_feature:
        prefix = 'train' if train_data else 'val'
        with open(os.path.join(opt.save_folder, 'features_supcon_{}.json'.format(prefix)), 'w') as f:
            json.dump(feature_dict, f)

    # anchor_list = []
    # alignment_list = []
    # for key in sorted(feature_dict.keys()):
    #     feature_list = np.array(feature_dict[key])
    #
    #     # find anchor
    #     min_dist = 1e10
    #     for feature in feature_list:
    #         dist = np.sum(np.abs(feature_list - feature))
    #         if dist <= min_dist:
    #             min_dist = dist
    #             anchor = feature
    #     pairwise_dist = np.linalg.norm(feature_list[:, None, :] - feature_list[None, :, :], axis=-1)
    #     alignment_list.append(np.mean(pairwise_dist))
    #     anchor_list.append(anchor)
    #
    # anchor_list = torch.Tensor(anchor_list).cuda()
    # anchor_list = torch.cat([anchor_list.unsqueeze(1), anchor_list.unsqueeze(1)], dim=1)
    # labels = torch.Tensor(list(sorted(feature_dict.keys()))).long().cuda()
    # anchor_loss = criterion(anchor_list, labels=labels)
    # np.set_printoptions(precision=3)
    # print("Alignment for each class:", np.array(alignment_list))
    # print("Class anchor Uniformity: {:.4f}".format(anchor_loss.detach().cpu().numpy()))
    # return losses.avg, alignment_meter.avg, uniformity_meter.avg, np.mean(alignment_list), anchor_loss.detach().cpu().numpy()
    return losses.avg, alignment_meter.avg, uniformity_meter.avg, 0, 0


def stage1(opt):
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if not opt.save_feature:
            loss, train_alignment, train_uniformity, train_class_alignment, train_class_uniformity = loop(train_loader, model, criterion, optimizer, epoch, opt, train=True, train_data=True, save_feature=opt.save_feature)
        else:
            with torch.no_grad():
                loss, train_alignment, train_uniformity, train_class_alignment, train_class_uniformity = loop(train_loader, model, criterion, optimizer, epoch, opt,
                                                               train=False, train_data=True, save_feature=opt.save_feature)
        with torch.no_grad():
            val_loss, val_alignment, val_uniformity, val_class_alignment, val_class_uniformity = loop(val_loader, model, criterion, optimizer, epoch, opt,
                                                           train=False, train_data=False, save_feature=opt.save_feature)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        if opt.save_feature:
            exit(0)

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('train_alignment', train_alignment, epoch)
        logger.log_value('train_class_alignment', train_class_alignment, epoch)
        logger.log_value('train_uniformity', train_uniformity, epoch)
        logger.log_value('train_class_uniformity', train_class_uniformity, epoch)
        logger.log_value('val_alignment', val_alignment, epoch)
        logger.log_value('val_class_alignment', val_class_alignment, epoch)
        logger.log_value('val_uniformity', val_uniformity, epoch)
        logger.log_value('val_class_uniformity', val_class_uniformity, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    opt = parse_option()
    opt = parse_option_stage1(opt)
    stage1(opt)
