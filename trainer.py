import pandas as pd
import joblib
from collections import OrderedDict
from bisect import bisect_right
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import torch
import os

from torch.optim import lr_scheduler
from tqdm import tqdm

from models.RConFormer import *
from utils import *
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def coo():
    global cuda_a
    cuda_a = 0
    return cuda_a


def load_data(args):
    if args['dataset']=='ARIL':
        # Linear interpolate train_data
        data_amp = sio.loadmat(args['dataset_path'] + '/linear_train_data.mat')
        train_data_amp = data_amp['train_data']
        train_data = train_data_amp
        train_activity_label = data_amp['train_label_activity']
        train_location_label = data_amp['train_label_location']
        train_label = np.concatenate((train_activity_label, train_location_label), 1)
        # tensor train_data
        train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_label = torch.from_numpy(train_label).type(torch.LongTensor)
        train_dataset = TensorDataset(train_data, train_label)

        # Linear interpolate test_data
        data_amp = sio.loadmat(args['dataset_path'] + '/linear_test_data.mat')
        test_data_amp = data_amp['test_data']
        test_data = test_data_amp
        test_activity_label = data_amp['test_label_activity']
        test_location_label = data_amp['test_label_location']
        test_label = np.concatenate((test_activity_label, test_location_label), 1)
        # tensor test_data
        test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
        test_label = torch.from_numpy(test_label).type(torch.LongTensor)
        test_dataset = TensorDataset(test_data, test_label)

        train_data = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)
        test_data = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=False)

    else:
        # dataset == 'UT-HAR':
        dataset = torch.load(args['dataset_path'])
        #aclist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        #print(len(dataset))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_data = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=1)
        test_data = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=1)

    return train_data, test_data


def _train(args, train_loader, model, criterion, optimizer):
    act_losses = AverageMeter()
    act_acc1s = AverageMeter()
    model.train()

    for i, (samples, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args['dataset'] == 'ARIL':
            samplesV = Variable(samples).cuda(coo())        # torch.Size([batchsize, channels, timestamps])
            labels_act = labels[:, 0].squeeze()
            labelsV_act = Variable(labels_act).cuda(coo())  # torch.Size([batchsize])
        else:
            labels_act = labels.long()
            samplesV, labelsV_act = samples.cuda(coo()), labels_act.cuda(coo())
            samplesV = torch.permute(samplesV, (0, 2, 1))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label_act = model(samplesV)
        loss = criterion(predict_label_act, labelsV_act)

        act_acc1, act_acc5 = accuracy(predict_label_act, labelsV_act, topk=(1, 5))

        act_losses.update(loss.item(), labels_act.size(0))
        act_acc1s.update(act_acc1.item(), labels_act.size(0))

        # compute gradient and do optimizing step
        loss.backward()
        if args['dataset'] == 'UT-HAR': torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


    log = OrderedDict([
        ('act_loss', act_losses.avg),
        ('act_acc', act_acc1s.avg),
    ])

    return log


def _validate(args, val_loader, model, criterion):
    act_losses = AverageMeter()
    act_acc1s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (samples, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):

            if args['dataset'] == 'ARIL':
                samplesV = Variable(samples).cuda(coo())  # torch.Size([batchsize, channels, timestamps])
                labels_act = labels[:, 0].squeeze()
                labelsV_act = Variable(labels_act).cuda(coo())  # torch.Size([batchsize])
            else:
                labels_act = labels.long()
                samplesV, labelsV_act = samples.cuda(coo()), labels_act.cuda(coo())
                samplesV = torch.permute(samplesV, (0, 2, 1))

            predict_label_act = model(samplesV)
            loss = criterion(predict_label_act, labelsV_act)
            act_acc1, act_acc5 = accuracy(predict_label_act, labelsV_act, topk=(1, 5))

            act_losses.update(loss.item(), labels_act.size(0))
            act_acc1s.update(act_acc1.item(), labels_act.size(0))

    log = OrderedDict([
        ('act_loss', act_losses.avg),
        ('act_acc', act_acc1s.avg),
    ])

    return log


def train(args):
    #args = parse_args()
    assert (
            args['dataset'] == 'ARIL' or args['dataset'] == 'UT-HAR'
    ), f"Warning: The dataset is supposed to be ARIL or UT-HAR !"

    best_acces = []
    for seed in args['seeds']:
        setup_seed(seed)

        if not os.path.exists('trained_models/%s/%s/%s-%s' % (args['dataset'], args['model'], args['model'], seed)):
            os.makedirs('trained_models/%s/%s/%s-%s' % (args['dataset'], args['model'], args['model'], seed))

        print('Config -----')

        for arg in args:
            print('%s: %s' % (arg, args[arg]))
        print('seed now: %s' % (seed))
        print('------------')

        with open('trained_models/%s/%s/%s-%s/args.txt' % (args['dataset'], args['model'], args['model'], seed),
                  'w') as f:
            for arg in args:
                print('%s: %s' % (arg, args[arg]), file=f)

        joblib.dump(args,
                    'trained_models/%s/%s/%s-%s/args.pkl' % (args['dataset'], args['model'], args['model'], seed))

        # data loading code
        train_data, val_data = load_data(args)

        # create model
        model = R_ConFormer(n_way=args['class_num'], input_shape=args['data_shape'], n_head=args['n_head'],
                            n_cnn_layers=4, n_encoder_layers=1, t_encoder=4, dim_projection=128, dim_feedforward=256)
        if torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=[0])
            model = model.cuda(coo())

        criterion = nn.CrossEntropyLoss().cuda(coo())
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

        # scheduler
        if args['dataset'] == 'UT-HAR':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(e) for e in args['milestones'].split(',')], gamma=args['gamma'])
        else:
            # ARIL: warmup + multistep
            scheduler = WarmupMultiStepLR(optimizer, milestones=[int(e) for e in args['milestones'].split(',')], gamma=args['gamma'],
                                          warmup_factor=0.1, warmup_iters=5, warmup_method='linear', last_epoch=-1)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'lr', 'act_loss', 'act_acc', 'val_act_loss', 'val_act_acc'])


        best_act_acc = 0
        for epoch in range(args['epochs']):
            print('Epoch [%d/%d]' % (epoch + 1, args['epochs']))
            print('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

            # train for one epoch
            train_log = _train(args, train_data, model, criterion, optimizer)
            # evaluate on validation set
            val_log = _validate(args, val_data, model, criterion)
            scheduler.step()

            print('act_loss %.4f - act_acc %.4f - val_act_loss %.4f '
                  '- val_act_acc %.4f ' % (
                      train_log['act_loss'], train_log['act_acc'],
                      val_log['act_loss'], val_log['act_acc']))

            tmp = pd.Series([
                epoch,
                scheduler.get_last_lr()[0],
                train_log['act_loss'],
                train_log['act_acc'],
                val_log['act_loss'],
                val_log['act_acc'],
            ], index=['epoch', 'lr', 'act_loss', 'act_acc', 'val_act_loss', 'val_act_acc'])

            log = log.append(tmp, ignore_index=True)
            log.to_csv('trained_models/%s/%s/%s-%s/log.csv' % (args['dataset'], args['model'], args['model'], seed),
                       index=False)

            if val_log['act_acc'] > best_act_acc:
                torch.save(model, 'trained_models/%s/%s/%s-%s/model.pth' % (
                    args['dataset'], args['model'], args['model'], seed))
                best_act_acc = val_log['act_acc']
                print("=> saved best model")

        # 记录所有seeds实验的输出
        acc_log = pd.DataFrame(index=[], columns=['seed', 'act_acc'])
        acc_tmp = pd.Series([
            seed,
            best_act_acc,
        ], index=['seed', 'act_acc'])
        acc_log = acc_log.append(acc_tmp, ignore_index=True)

        csv_head = False
        if not os.path.exists('trained_models/%s/%s-log.csv' % (args['dataset'], args['model'])): csv_head = True
        if csv_head:
            acc_log.to_csv(
                'trained_models/%s/%s-log.csv' % (args['dataset'], args['model']), index=False)
        else:
            acc_log.to_csv(
                'trained_models/%s/%s-log.csv' % (args['dataset'], args['model']), mode='a', header=False, index=False)


        print("best acc: %.4f " % best_act_acc)
        best_acces.append(best_act_acc)

    best_acces = torch.tensor(best_acces)
    avg_acc = torch.mean(best_acces, dtype=float)
    print("best acc of all experiments:")
    print(best_acces)
    print("average acc of all experiments: %.4f " % avg_acc)






