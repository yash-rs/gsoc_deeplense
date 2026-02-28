import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import logging
import random

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from networks import LensCNN
from dataset import LensDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class CNNTrainer:
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration dictionary.
        """
        self.config = config
        self.seed = self.config["experiment"]["seed"]
        set_seed(self.seed)

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
        self.run_name = self.config["experiment"]["run_name"]
        self.base_output_dir = self.config["experiment"]["output_dir"]
        self.run_dir = os.path.join(self.base_output_dir, self.run_name)
        self.plots_dir = os.path.join(self.run_dir, "plots")

        os.makedirs(self.plots_dir, exist_ok=True)

        ## Logging ##
        log_file = os.path.join(self.run_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger()

        self.setup()

    def setup(self):
        """
        """
        self.device = torch.device(
        "cuda" if torch.cuda.is_available() and self.config["device"]["use_cuda"]
        else "cpu"
        )
        self.logger.info("Initializing and setting up training...")
        self.logger.info(f"Configs used: {self.config}")

        # -------------------------
        # Datasets
        # -------------------------
        train_dataset = LensDataset(
            root_dir=self.config["data"]["root_dir"],
            split="train"
        )

        val_dataset = LensDataset(
            root_dir=self.config["data"]["root_dir"],
            split="val"
        )

        # -------------------------
        # Dataloaders
        # -------------------------
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"]
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"]
        )

        # -------------------------
        # Model
        # -------------------------
        self.model = LensCNN(
            in_channels=1,
            num_classes=self.config["model"]["num_classes"]
        ).to(self.device)

        # -------------------------
        # Loss
        # -------------------------
        self.criterion = nn.CrossEntropyLoss()

        # -------------------------
        # Optimizer
        # -------------------------
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

    def train(self):
        """
        Training loop

        """
        self.logger.info("Starting training...")
        num_epochs = self.config["training"]["epochs"]
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # ---------------------
            # TRAINING
            # ---------------------
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            self.train_losses = []
            self.val_losses = []

            for images, labels in self.train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Stats
                train_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss /= total
            self.train_losses.append(train_loss)
            train_acc = correct / total

            # ---------------------
            # VALIDATION
            # ---------------------
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_loss /= total
            self.val_losses.append(val_loss)
            val_acc = correct / total

            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            self._compute_and_plot_roc(epoch)

            # ---------------------
            # Save Best Model
            # ---------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(),
                        os.path.join(self.run_dir, "best_model.pth"))

        self._compute_and_plot_roc(epoch=None)
        self._plot_and_save_loss(num_epochs)
        

    def _compute_and_plot_roc(self, epoch=None):
        """
        Computes multi-class ROC (one-vs-rest) on validation set
        and saves plot.
        """

        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)

                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        num_classes = self.config["model"]["num_classes"]

        # Binarize labels
        y_true = label_binarize(all_labels, classes=list(range(num_classes)))

        plt.figure()

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-Class ROC Curve")
        plt.legend()

        if epoch is not None:
            save_path = os.path.join(self.plots_dir, f"roc_epoch_{epoch+1}.png")
        else:
            save_path = os.path.join(self.run_dir, "roc_final.png")

        plt.savefig(save_path)
        plt.show()
        plt.close()

    def _plot_and_save_loss(self, num_epochs):
        plt.figure()

        epochs = range(1, num_epochs + 1)

        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        save_path = os.path.join(self.run_dir, "loss_curve.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def inference(self, dataloader):
        """
        Run inference:
        - Model evaluation mode
        - Return predictions and probabilities
        """
        pass