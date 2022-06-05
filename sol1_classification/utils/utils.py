import numpy as np
import torch
def get_result(model,test_loader,device):
    y_test = []
    y_pred = []
    for data, target in test_loader:
        data = data.to(device)
        output = model(data).to("cpu")
        _, pred = torch.max(output, 1)
        pred = pred.data.numpy()
        target = target.data.numpy()
        y_pred.append(pred.flatten()[:])
        y_test.append(target.flatten()[:])
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    return y_test, y_pred