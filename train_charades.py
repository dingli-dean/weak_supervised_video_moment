from __future__ import print_function
import os
import pickle

import numpy
from data_charades import get_loaders
import time
import numpy as np
from vocab import Vocabulary
import torch
from torch.nn.utils.clip_grad import clip_grad_norm
from model_charades import VSE, order_sim
# from config import Config
from new_config import config
from collections import OrderedDict
import pandas
import logging
import evaluation_charades as evaluation
# os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_devices
# import ipdb

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    # attn_weights =
    for i, (images, captions, lengths, lengths_img, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, attn_weight_s = model.forward_emb(images, captions, lengths, lengths_img, volatile=True)

        if (attn_weight_s.size(1) < 10):
            attn_weight = torch.zeros(attn_weight_s.size(0), 10, attn_weight_s.size(2))
            attn_weight[:, 0:attn_weight_s.size(1), :] = attn_weight_s
        else:
            attn_weight = attn_weight_s

        batch_length = attn_weight.size(0)
        attn_weight = torch.squeeze(attn_weight)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            attention_index = np.zeros((len(data_loader.dataset), 10))
            rank1_ind = np.zeros((len(data_loader.dataset)))
            lengths_all = np.zeros((len(data_loader.dataset)))

        attn_index = np.zeros((batch_length, 10))  # Rank 1 to 10
        rank_att1 = np.zeros(batch_length)
        temp = attn_weight.data.cpu().numpy().copy()
        for k in range(batch_length):
            att_weight = temp[k, :]
            sc_ind = numpy.argsort(-att_weight)
            rank_att1[k] = sc_ind[0]
            attn_index[k, :] = sc_ind[0:10]

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        attention_index[ids] = attn_index
        lengths_all[ids] = lengths_img
        rank1_ind[ids] = rank_att1

        # measure accuracy and record loss
        model.train_emb(img_emb, cap_emb)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs, attention_index, lengths_all


# def cIoU_old(a,b,prec):
#    return np.around(1.0*(min(a[1], b[1])-max(a[0], b[0]))/(max(a[1], b[1])-min(a[0], b[0])),decimals=prec)


def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection) / union


def train(config, data_path=None, split='train', fold5=False):
    """
    Evaluate a trained model.
    """
    # load model and options
    # checkpoint = torch.load(model_path)
    # opt = checkpoint['opt']
    opt = config

    if not os.path.exists(opt.resume):
        os.mkdir(opt.resume)


    if data_path is not None:
        opt.data_path = data_path
    # opt.vocab_path = "./vocab/"
    # load vocabulary
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, 'vocab.pkl'), 'rb'))

    opt.vocab_size = len(vocab)

    batch_time = AverageMeter()
    # train_logger = LogCollector()
    # train_logger = logging.getLogger(__name__)

    # construct model
    model = VSE(opt)
    model.train_start()
    model.logger = logging.getLogger(__name__)
    # print(model.state_dict())

    model.logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(opt.resume + "/train_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    model.logger.addHandler(handler)

    # load model state
    # model.load_state_dict(checkpoint['model'])
    # print(opt)

    ####### input video files
    # path = os.path.join(opt.data_path, opt.data_name) + "/Caption/charades_" + str(split) + ".csv"
    # df = pandas.read_csv(open(path, 'rb'))
    # # columns=df.columns
    # inds = df['video']
    # desc = df['description']

    model.logger.info(opt.__dict__)
    model.logger.info('Loading dataset')

    if opt.no_val:
        train_loader = get_loaders(opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)
    else:
        train_loader, val_loader = get_loaders(opt.data_name, vocab, opt.crop_size,
                                      opt.batch_size, opt.workers, opt)


    model.logger.info('Start Training')



    for epoch in range(opt.train_epoches):

        if model.Eiters % opt.TRAIN.save_per_epoch == 0 and model.Eiters != 0:
            torch.save({'epoch': model.Eiters,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict(),
                        'opt': opt
                        }, os.path.join(opt.resume, '%d'% model.Eiters + '.pth.tar'))


            if opt.train_test:
                test_result = evaluation.evalrank(model_path=os.path.join(opt.resume, '%d'% model.Eiters + '.pth.tar'),
                                    data_path=opt.data_path, split="test")
                model.logger.info('Test Result' + '\n' + test_result)


        model.Eiters += 1

        for i, (images, captions, lengths, lengths_img, ids) in enumerate(train_loader):

            # compute the embeddings
            if opt.TRAIN.back_caption:
                img_emb, cap_emb, attn_weights, b_cap = model.forward_emb_with_back_caption(images, captions,lengths, lengths_img, opt)
                loss, contrast_loss, bcap_loss = model.forward_constrative_caption_loss(img_emb, cap_emb, b_cap, captions, opt.TRAIN.bploss_lambda)
            else:
                img_emb, cap_emb, attn_weights, _ = model.forward_emb(images, captions, lengths, lengths_img)
                # measure accuracy and record loss
                # self.optimizer.zero_grad()
                loss = model.forward_loss(img_emb, cap_emb)

            if i % 10 == 0 or i == (len(train_loader)-1):
                if opt.TRAIN.back_caption:
                    model.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' +
                                      'Epoch ' + str(model.Eiters) + '' + '---' +
                                      'Training[{}/{}]'.format(i, len(train_loader)) +
                                      '---' + 'loss %.4f' % (loss / opt.batch_size) +
                                      '---' + 'contrast_loss %.4f' % (contrast_loss / opt.batch_size) +
                                      '---' + 'bcap_loss %.4f' % (bcap_loss / opt.batch_size))
                else:
                    model.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' +
                                      'Epoch ' + str(model.Eiters) + '' + '---' * 2 +
                                      'Training[{}/{}]'.format(i, len(train_loader)) +
                                      '---' * 2 + 'loss %.4f' % (loss / opt.batch_size))

            # compute gradient and do SGD step
            loss.backward()
            if model.grad_clip > 0:
                clip_grad_norm(model.params, model.grad_clip)
            model.optimizer.step()

    model.logger.info('Epoch %d Training Finished' % opt.train_epoches)
    model.logger.info(opt.__dict__)

if __name__ == '__main__':

    train(config)
