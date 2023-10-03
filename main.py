import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchsummary import summary
import numpy as np
import time
import os
from util.masks import get_mask
from util.data import load_data, get_data_loader
from transformer import RTIDS_Transformer, IDS_Encoder_Only
from sklearn.metrics import classification_report

def accuracy_per_class(model, loader):
    model.eval()
    model.to("cuda")
    correct = np.zeros(15, dtype=np.int64)
    wrong = np.zeros(15, dtype=np.int64)
    with torch.no_grad():
        for src, trg, at_class in loader:
            src, trg = src.to("cuda"), trg.to("cuda")
            output = model(src)            
            for label, pred, at_c in zip(trg, output, at_class.cpu().numpy()):
                at_c = int(at_c)
                if torch.eq(torch.argmax(pred),torch.argmax(label)):
                    correct[at_c] += 1
                else:
                    wrong[at_c] += 1
    accuracy = 100. * correct / (correct + wrong)
    return accuracy

def eval_model(model, loader):
    model.cuda()
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            losses.append(F.cross_entropy(output, target).item())
            correct += torch.eq(torch.argmax(output, dim=1),torch.argmax(target, dim=1)).cpu().sum().item()
    eval_loss = float(np.mean(losses))
    eval_acc = 100. * correct / len(loader.dataset)
    print("Loss:", eval_loss, "Accuracy:", eval_acc)
    return eval_loss, eval_acc

def train_model(model, opt, epochs, data, eval_data, path, print_every=100):
    model.cuda()
    
    pretrained_path = "pretrained"
    top_acc = 0.

    if os.path.exists(pretrained_path + "/" + path):
        print("Loading Pretrained Model")
        state = torch.load(pretrained_path + "/" + path)
        model.load_state_dict(state["model_state_dict"])
        start_epoch = state["epoch"] + 1
        losses = state["ep_loss"]
        accs = state["ep_acc"]
        top_acc = max(accs)
    else:
        start_epoch = 0
        losses, accs = [], []
        try:
            os.mkdir(pretrained_path)
        except OSError as error:
            pass 


    start = time.time()
    temp = start
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(data):
            src,trg,_ = batch
            src,trg = src.cuda(), trg.cuda()
            
            trg_mask = get_mask(78)
            
            preds = model(src, trg_mask)            
            opt.zero_grad()
            loss = F.cross_entropy(preds, trg)
            loss.backward()
            opt.step()
            
            total_loss += loss.data 
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, \
                %ds per %d iters" % ((time.time() - start) // 60, 
                epoch + 1, i + 1, loss_avg, time.time() - temp, 
                print_every))
                total_loss = 0
                temp = time.time()
        ep_loss, ep_acc = eval_model(model,eval_data)

        losses.append(ep_loss)
        accs.append(ep_acc)
        if ep_acc > top_acc:
            top_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch
            }
            torch.save(top_state, pretrained_path + "/max_" + path)
        state = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'ep_loss': losses,
                'ep_acc': accs
            }
        torch.save(state, pretrained_path + "/" + path)

def main():
    learning_rate = 5e-4
    batch_size = 128
    epochs = 27
    dropout_rate = 0.5
    d_model = 32
    heads = 8
    N = 6
    trg_vocab = 2
    
    
    train_data, val_data = load_data()
    
    train_loader = get_data_loader(train_data, batch_size)
    val_loader = get_data_loader(val_data, batch_size)

    model = RTIDS_Transformer(trg_vocab, d_model, N, heads, dropout_rate)
    save_path = "RTIDS_pretrained_copy.pt"
    # model = IDS_Encoder_Only(trg_vocab, d_model, N, heads, dropout_rate)
    # save_path = "pretrained_enc.pt"

    pretrained_path = "pretrained"

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    summary(model)
    
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_model(model, optim, epochs, train_loader, val_loader, save_path)
    # if os.path.exists(pretrained_path + "/" + save_path):
    #     print("Loading Pretrained Model")
    #     state = torch.load(pretrained_path + "/" + save_path)
    #     model.load_state_dict(state["model_state_dict"])
    # accuracy_per_class(model, val_loader)

if __name__ == "__main__":
    main()