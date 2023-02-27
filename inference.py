import torch
from matplotlib import pyplot as plt
from utils import *


def inference(dataset, input, model_path):
    assert (
            dataset == 'ARIL' or dataset == 'UT-HAR'
    ), f"Warning: The dataset is supposed to be ARIL or UT-HAR !"

    model = torch.load(model_path)
    device = "cuda:0"
    model = model.to(device)

    # The form of input should be [timestamps, channels]
    X = torch.load(input)
    plt.plot(X)
    plt.savefig(str(input)[:-3])
    plt.show()

    with torch.no_grad():
        X = torch.from_numpy(X).float()
        X = torch.unsqueeze(X.permute(1, 0), 0)
        outputs = torch.softmax(model(X), 1)
        pred = torch.argmax(outputs)

    if dataset == 'ARIL':
        AClist = ['up', 'down', 'left', 'right', 'circle', 'cross']
    else:
        AClist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']

    print("The action of input wifi signal is \"" + str(AClist[pred] + "\" !"))

    return outputs


if __name__ == '__main__':

    inference("ARIL", "data/samples/sample_ARIL.pt", "trained_models/ARIL/Act_RConFormer/Act_RConFormer-5/model.pth")
    #inference("UT-HAR", "data/samples/sample_UTHAR.pt", "trained_models/UT-HAR/RConFormer/RConFormer-5/model.pth")



