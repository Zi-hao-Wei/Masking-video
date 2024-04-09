import transformers

import torch
from trainer_MPII import Multi_Trainer_dist_MPII
from MPII import MPIIDataset
from torch.utils.data import DataLoader
from model.video_transformer_flip import SpaceTimeTransformer
from model.naive_model import FrozenInTime

from model.loss import NormSoftmaxLoss



def run():
   
    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                                           TOKENIZERS_PARALLELISM=False)

    # setup data_loader instances
    data_loader = DataLoader(MPIIDataset(), batch_size=8, shuffle=True,drop_last=True) #batch_size=4
    
    model = FrozenInTime().cuda()

    loss = NormSoftmaxLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    writer = None
 
    trainer = Multi_Trainer_dist_MPII(model, loss, optimizer,
                      data_loader=data_loader,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=500000)
    
    trainer.train()
    

if __name__ == '__main__':

    run()