import glob
from xmlrpc.client import boolean
import pandas as pd
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision import models
from torch import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='../dataset/test', help='dataset root  folder name')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='model path')
    parser.add_argument('--recursive', type=boolean, default=1, help='1 for true, 0 for false')
    parser.add_argument('--output_path', type=str, default='output.csv', help='output csv path')
    opt = parser.parse_args()

    
    folder_path = opt.folder_path
    device = opt.device
    model_path = opt.model_path
    recur = opt.recursive
    output_path = opt.output_path
    

    classes = ['bird','cat']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    checkpoint = torch.load(model_path, map_location=device)
    model_ft.load_state_dict(checkpoint['model_state_dict'])
    model_ft.eval()
    model_ft.to(device)

    img_list = glob.glob(str(folder_path)+'/**/*.jpeg', recursive=recur)
    result =[]
    for i in img_list:
        img_file = Image.open(i)
        input = transformations(img_file).to(device)
        output = model_ft(input.unsqueeze(0)).to(device)
        _, pred = torch.max(output, 1)
        result.append([i, classes[pred.item()]])
    pd.DataFrame(result, columns=['files', 'target']).to_csv(output_path, index=False)
    print('done')


    
