import torch.backends.cudnn as cudnn
from torch.autograd import Variable
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


def coo():
    global cuda_a
    cuda_a = 0
    return cuda_a


def load_data(args):
    if args['dataset'] == 'ARIL':
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
    else:
        # dataset == 'UT-HAR':
        dataset = torch.load(args['dataset_path'])
        #aclist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        #print(len(dataset))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return test_dataset


def _evaluation(args, val_loader, model, criterion):
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


def evaluation(config_file, model_path):
    args = load_json(config_file)
    assert (
            args['dataset'] == 'ARIL' or args['dataset'] == 'UT-HAR'
    ), f"Warning: The dataset is supposed to be ARIL or UT-HAR !"
    setup_seed(args['seeds'][0])     # default：The first one of the seed list.

    test_dataset = load_data(args)
    test_data = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=False)

    # load model
    model = torch.load(model_path)
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0])
        model = model.cuda(coo())
    criterion = nn.CrossEntropyLoss().cuda(coo())

    val_log = _evaluation(args, test_data, model, criterion)

    print('val_act_loss %.4f - val_act_acc %.4f ' % (val_log['act_loss'], val_log['act_acc']))


if __name__ == '__main__':
    #evaluation("./configs/ARIL.json", "trained_models/ARIL/Act_RConFormer/Act_RConFormer-5/model.pth")
    evaluation("./configs/UT-HAR.json", "trained_models/UT-HAR/RConFormer/RConFormer-5/model.pth")



