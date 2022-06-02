import torch
import matplotlib.pyplot as plt

def visualization(dtest):
    keys = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    model.eval()
    for (x, y) in dtest:
        with torch.no_grad():
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y1, y2, y3, y4, y5 = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]

            pred1, pred2, pred3, pred4, pred5 = model(x)
            pred1 = torch.argmax(pred1, -1)
            pred2 = torch.argmax(pred2, -1)
            pred3 = torch.argmax(pred3, -1)
            pred4 = torch.argmax(pred4, -1)
            pred5 = torch.argmax(pred5, -1)

            y1 = list(map(lambda x: keys[x], y1))
            y2 = list(map(lambda x: keys[x], y2))
            y3 = list(map(lambda x: keys[x], y3))
            y4 = list(map(lambda x: keys[x], y4))
            y5 = list(map(lambda x: keys[x], y5))

            pred1 = list(map(lambda x: keys[x], pred1))
            pred2 = list(map(lambda x: keys[x], pred2))
            pred3 = list(map(lambda x: keys[x], pred3))
            pred4 = list(map(lambda x: keys[x], pred4))
            pred5 = list(map(lambda x: keys[x], pred5))


            for idx in range(len(y1)):
                true_str = f'{y1[idx]} {y2[idx]} {y3[idx]} {y4[idx]} {y5[idx]}'
                pred_str = f'{pred1[idx]} {pred2[idx]} {pred3[idx]} {pred4[idx]} {pred5[idx]}'
                plt.figure(figsize=(50,50))
                plt.subplot(8, 8, idx+1)
                plt.title(f'{true_str} / {pred_str}')
                plt.imshow(x[idx].detach().cpu().permute(1, 2, 0))
            break