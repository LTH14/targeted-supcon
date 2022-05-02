from imagenet_inat.models.ResNetFeature import *
from imagenet_inat.utils import *
from os import path


def create_model(use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(BasicBlock, [1, 1, 1, 1], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None)

    if not test:
        if stage1_weights:
            assert dataset
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            if log_dir is not None:
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading weights from %s' % weight_dir)
            resnet10 = init_weights(model=resnet10,
                                    weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet10
