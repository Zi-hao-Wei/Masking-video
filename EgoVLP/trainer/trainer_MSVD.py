import numpy as np
import torch

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class Multi_Trainer_dist_MSVD:
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self,  model, loss,  optimizer, data_loader,batch_size,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):

        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.writer = writer

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        self.model.train()
        total_loss = [0] * len(self.data_loader)
 
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)

        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):

                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                  truncation=True)
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)
                
                # print(data["text"].shape)
                # print(data["video"].shape)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    text_embeds, video_embeds = self.model(data)
                    output = sim_matrix(text_embeds, video_embeds)
                    loss = self.loss(output)

                loss.backward()

                self.optimizer.step()

               
                total_loss[dl_idx] += loss.detach().item()

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return
    
    def train(self):
        """
        Full training logic
        """

        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)
        torch.save(self.model.state_dict(), 'Final.pth')
