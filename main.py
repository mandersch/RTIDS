from torch import nn
import torch
import util
from torch.nn import functional as F
from util.masks import get_mask
from util.data import load_data, get_data_loader
from transformer import RTIDS_Transformer
import time
from torchsummary import summary
import os

def train_model(model, opt, epochs, data, mask, print_every=100):
    model.cuda()
    model.train()
    start = time.time()
    temp = start
    
    total_loss = 0
    
    for epoch in range(epochs):
       
        for i, batch in enumerate(data):
            src,trg = batch
            src,trg = src.cuda(), trg.cuda()
            src,trg = src.type(torch.LongTensor).cuda(), trg.type(torch.LongTensor).cuda()
            
            trg_mask = mask
            
            preds = model(src, trg, trg_mask)
            
            opt.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), trg)
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
    os.makedirs("pretrained")
    state = {
            'model_state_dict': model.state_dict(),
        }
    torch.save(state, "pretrained/pretrained.pt")

def main():
    learning_rate = 5e-4
    batch_size = 128
    epochs = 1
    dropout_rate = 0.5
    d_model = 32
    heads = 8
    N = 6
    src_vocab = 10000
    trg_vocab = 15
    mask = get_mask(src_vocab)
    train_data, val_data = load_data()
    train_loader = get_data_loader(train_data, batch_size)

    model = RTIDS_Transformer(src_vocab, src_vocab, d_model, N, heads, dropout_rate)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    summary(model)
    
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_model(model, optim, epochs, train_loader, mask)

if __name__ == "__main__":
    main()