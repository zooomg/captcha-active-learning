import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def evaluate(model, test_dataloader, experiment, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    pred_list = [[] for i in range(5)]
    true_list = [[] for i in range(5)]
    current_losses = []
    
    device = f'cuda:{args.gpu_number}'

    for (x, y) in tqdm(test_dataloader, desc="eval"):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            y1, y2, y3, y4, y5 = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]

            pred1, pred2, pred3, pred4, pred5 = model(x)
            
            loss1 = criterion(pred1, y1)
            loss2 = criterion(pred2, y2)
            loss3 = criterion(pred3, y3)
            loss4 = criterion(pred4, y4)
            loss5 = criterion(pred5, y5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            current_losses.append([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()])
            
            
            pred1 = torch.argmax(pred1, -1)
            pred2 = torch.argmax(pred2, -1)
            pred3 = torch.argmax(pred3, -1)
            pred4 = torch.argmax(pred4, -1)
            pred5 = torch.argmax(pred5, -1)
            

            pred_list[0] += pred1.detach().cpu().tolist()
            pred_list[1] += pred2.detach().cpu().tolist()
            pred_list[2] += pred3.detach().cpu().tolist()
            pred_list[3] += pred4.detach().cpu().tolist()
            pred_list[4] += pred5.detach().cpu().tolist()

            true_list[0] += y1.detach().cpu().tolist()
            true_list[1] += y2.detach().cpu().tolist()
            true_list[2] += y3.detach().cpu().tolist()
            true_list[3] += y4.detach().cpu().tolist()
            true_list[4] += y5.detach().cpu().tolist()
            
    pred_list = np.array(pred_list)
    true_list = np.array(true_list)

    current_losses = np.mean(current_losses, axis=0)
    current_loss = np.sum(current_losses)
    
    accuracy = np.sum(pred_list==true_list, axis=1)/true_list.shape[-1]
    mean_accuracy = np.mean(accuracy).item()
    
    print('Initial Test')
    print(f"loss={current_loss:.4f}")
    print(f"losses={current_losses}")
    print(f"mean_accuracy={mean_accuracy:.4f}")
    print(f"accuracy={accuracy}")

    if args.wandb:
        experiment.log(
            {
                "test_loss_1": current_losses[0], 
                "test_loss_2": current_losses[1], 
                "test_loss_3": current_losses[2], 
                "test_loss_4": current_losses[3], 
                "test_loss_5": current_losses[4], 
                "test_accuracy_1": accuracy[0],
                "test_accuracy_2": accuracy[1],
                "test_accuracy_3": accuracy[2],
                "test_accuracy_4": accuracy[3],
                "test_accuracy_5": accuracy[4],
            }
        )
    # return current_losses, current_loss, accuracy, mean_accuracy