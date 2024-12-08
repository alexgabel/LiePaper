import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
         # Number of points in validation set
        self.v_num = len(self.valid_data_loader.sampler)
        # Initialize t_history to be a numpy array of zeros:
        self.t_history = np.zeros((self.epochs, self.v_num))

        #tf_range = self.config["data_loader"]["args"]["tf_range"]

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # Before training, calibrate the model by running the training
        # set through the model once:
        #### TODO ####

    # def loss_fn(self, output, target, data, inputtnet, combined):
    #     # Get number of channels from config
    #     channels = self.config['arch']['args']['channels']
    #     bs = data.size(0)
    #     batch_tG = torch.einsum('i, mn -> imn', self.model.t(inputtnet).squeeze(), self.model.G)
    #     exp_tG = torch.matrix_exp(batch_tG)
    #     loss = self.criterion(output, target) \
    #             + self.config['lambda_recon'] * ( self.criterion(self.model.normal(data),data) \
    #                 + self.criterion(self.model.normal(target),target) ) \
    #             + self.config['lambda_z'] * self.criterion(self.model.encoder(target.view(bs, -1)).view(bs,channels,-1), #.mean(dim=2),
    #                                                         torch.einsum('ica, iba -> icb', 
    #                                                                     self.model.encoder(data.view(bs, -1)).view(bs,channels,-1), #.mean(dim=2),
    #                                                                     exp_tG
    #                                                                     )
    #                                                         ) \
    #             + self.config['lambda_lasso'] * torch.norm(self.model.a, p=1) \
    #             + self.config['lambda_a'] * self.criterion(*self.model.taylor_loss(combined))
    #     return loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            ##### T-NET #####
            # Combine the data and target tensors into a single tensor:

            combined = torch.cat((data, target), 1)


            xtnet, ytnet = torch.split(combined, 1, dim=1)

            # Save size before flattening input:
            data_shape = xtnet.size()

            # Flatten x and y:
            xtnet = xtnet.view(xtnet.size(0), -1)
            ytnet = ytnet.view(ytnet.size(0), -1)   

            inputtnet = torch.cat((xtnet,ytnet), dim=1)
            #################

            self.optimizer.zero_grad()
            output, exptG, t = self.model(combined) # Change to self.model(data) for non-T-Net

            loss = self.criterion(output, target, data, exptG, combined,
                                  self.config, self.model)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            t = t.detach().cpu().numpy()
            self.writer.add_histogram('t', t, epoch)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, target, data, exptG, 
                                                                combined, self.config, self.model).item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data[:8*5].cpu(), nrow=8, normalize=True))
                self.writer.add_image('output', make_grid(output[:8*5].cpu(), nrow=8, normalize=True))
                self.writer.add_image('target', make_grid(target[:8*5].cpu(), nrow=8, normalize=True))
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                ###### T-NET ######
                # Combine the data and target tensors into a single tensor:
                combined = torch.cat((data, target), 1)

                xtnet, ytnet = torch.split(combined, 1, dim=1)

                # Save size before flattening input:
                data_shape = xtnet.size()

                # Flatten x and y:
                xtnet = xtnet.view(xtnet.size(0), -1)
                ytnet = ytnet.view(ytnet.size(0), -1)   

                inputtnet = torch.cat((xtnet,ytnet), dim=1)
                ###################

                output, exptG , t = self.model(combined) # Change to self.model(data) for no T-NET
                loss = self.criterion(output, target, data, exptG, combined, self.config, self.model)
                if batch_idx == 0:
                    print('alpha: ', self.model.a/torch.norm(self.model.a, p=2))
                # Logging t values to tensorboard using histogram:
                ########################################
                t_log = t.detach().cpu().numpy()
                self.writer.add_histogram('t', t_log, epoch)
                # Accumulate t values in t_history for single epoch:
                self.t_history[epoch-1, batch_idx*data.shape[0]:(batch_idx+1)*data.shape[0]] = t_log
                ########################################
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, data, exptG, 
                                                                combined ,self.config, self.model))
                self.writer.add_image('input', make_grid(data[:8*5].cpu(), nrow=8, normalize=True))
                self.writer.add_image('output', make_grid(output[:8*5].cpu(), nrow=8, normalize=True))
                self.writer.add_image('target', make_grid(target[:8*5].cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
