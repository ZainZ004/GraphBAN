import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from domain_adaptator import ReverseLayerF
from tqdm import tqdm
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optim, device, test_dataloader, opt_da=None, discriminator=None, experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.experiment = experiment
        self.config = config

        # Domain Adaptation setup
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self._setup_domain_adaptation(config, discriminator)

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}

    def _setup_domain_adaptation(self, config, discriminator):
        self.da_method = config["DA"]["METHOD"]
        self.domain_dmm = discriminator
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]

        if config["DA"]["RANDOM_LAYER"]:
            if config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = nn.Linear(
                    in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                    out_features=config["DA"]["RANDOM_DIM"],
                    bias=False
                ).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
        else:
            self.random_layer = None

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.batch_size) / (non_init_epoch * self.batch_size)
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        y_pred = self.test(dataloader="test")
        return y_pred

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy) # type: ignore
        return entropy_w

    def test(self, dataloader="test"):
        y_pred = []
        if dataloader == "test":
            data_loader = self.test_dataloader
        else:
            raise ValueError(f"Invalid dataloader key: {dataloader}")

        with torch.no_grad():
            self.model.eval()
            for v_d, sm, v_p, esm in data_loader:
                sm = torch.tensor(sm, dtype=torch.float32).reshape(sm.shape[0], 1, 384).to(self.device)
                esm = torch.tensor(esm, dtype=torch.float32).reshape(esm.shape[0], 1, 1280).to(self.device)
                v_d, v_p = v_d.to(self.device), v_p.to(self.device)

                _, _, _, score = self.model(v_d, sm, v_p, esm, self.device)

                if self.n_class == 1:
                    m = nn.Sigmoid()
                    n = torch.squeeze(m(score), 1)
                else:
                    n = cross_entropy_logits(score)

                y_pred.extend(n.cpu().tolist())

        return y_pred
