import os
import numpy as np
import matplotlib as plt
import sklearn

import torch


class CNNTrainer:
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration dictionary.
        """
        self.config = config

        # Device
        self.device = None

        # Data
        self.train_loader = None
        self.val_loader = None

        # Model components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def setup(self):
        """
        Setup:
        - Device
        - Dataset & Dataloaders
        - Model
        - Loss function
        - Optimizer
        - Scheduler (optional)
        """
        pass

    def train(self):
        """
        Training loop:
        - Forward pass
        - Loss computation
        - Backprop
        - Validation
        - Metric computation (ROC/AUC later)
        """
        pass

    def inference(self, dataloader):
        """
        Run inference:
        - Model evaluation mode
        - Return predictions and probabilities
        """
        pass