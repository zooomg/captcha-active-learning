import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np

def train(model, train_dataloader, experiment, desc, args):
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    model_path = f'{args.chkt_filename}_{args.uncertain_criteria}_{args.digit_compression}.pt'
    device = f'cuda:{args.gpu_number}'
    
    min_loss = 20.
    stop_cnt = 0
    for e in range(args.epochs):
        start_time = time.time()
        current_losses = []

        for (x, y) in tqdm(train_dataloader, desc):
            optimizer.zero_grad()
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
            loss.backward()
            optimizer.step()

        current_losses = np.mean(current_losses, axis=0)
        current_loss = np.sum(current_losses)

        if current_loss < min_loss:
            min_loss = current_loss
            torch.save(model.state_dict(), model_path)

        # ipd.clear_output(wait=True)
        print(f"{e+1}/{args.epochs}, {time.time()-start_time:.2f} sec/epoch")
        print(f"current loss={current_loss:.4f}")
        print(f"current losses={current_losses}")
        if args.wandb:
            experiment.log(
                {
                    "train_loss_1": current_losses[0],
                    "train_loss_2": current_losses[1],
                    "train_loss_3": current_losses[2],
                    "train_loss_4": current_losses[3],
                    "train_loss_5": current_losses[4],
                }
            )
        # plt.figure(figsize=(20,1),dpi=120)
        # plt.scatter(np.arange(len(loss_history)), loss_history, label='train')
        # plt.legend(loc=1)
        # plt.show()
