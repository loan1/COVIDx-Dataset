import torch

from PIL import Image


def test_loop(model_ft, device, test_dataloader):
    with torch.no_grad():
        y_true = []
        y_pred = []
        model_ft.to(device)
        model_ft.eval()
        for data, target in test_dataloader:
            batch_size = data.size(0)
            data = data.to(device)
            target = target.to(device)
            output = model_ft(data)
            _,pred = torch.max(output, 1)
            y_true += target.tolist()
            y_pred += pred.tolist()
    return y_true, y_pred

def img_transform(path_img, test_transform):
    img = Image.open(path_img)
    imagetensor = test_transform(img).cuda()
    return imagetensor


            