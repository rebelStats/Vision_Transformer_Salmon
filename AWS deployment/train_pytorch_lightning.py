# Based on https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb

# Updates:
#    _ Implement SageMaker Script Mode to run on Hugging Face Deep Learning Container
#    - Install PyTorch Lightning
#    - Read number of classes from training set

import sys, argparse, subprocess, os

import torch
import torch.nn as nn

from transformers import ViTModel, AdamW
from datasets import load_from_disk

def pip_install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package, "--upgrade"])

pip_install("pytorch_lightning")

from pytorch_lightning import LightningModule, Trainer

class ViTForImageClassification(LightningModule):
    def __init__(self, model_name, labels):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, labels)
        self.model_name = model_name
        self.labels = labels

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)
        return logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['label']
        logits = self(pixel_values)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("training_loss", loss, on_epoch=True)
        self.log("training_accuracy", accuracy, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader
    
    def test_dataloader(self):
        return test_dataloader
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model-name', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--train-batch-size', type=int, default=10)
    parser.add_argument('--eval-batch-size', type=int, default=4)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n-gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--valid-dir', type=str, default=os.environ['SM_CHANNEL_VALID'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    print("Received args: ", args)
    
    # load datasets
    print("Loading data sets...")
    train_dataset = load_from_disk(args.train_dir)
    valid_dataset = load_from_disk(args.valid_dir)
    test_dataset  = load_from_disk(args.test_dir)
    
    train_dataset.set_format('torch', columns=['pixel_values', 'label'])
    valid_dataset.set_format('torch', columns=['pixel_values', 'label'])
    test_dataset.set_format('torch', columns=['pixel_values', 'label'])
    
    print(train_dataset)
    print(valid_dataset)
    print(test_dataset)
    
    num_classes = train_dataset.features["label"].num_classes

    print("Building data loaders...")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=8)
    val_dataloader   = torch.utils.data.DataLoader(valid_dataset, batch_size=args.eval_batch_size, num_workers=8)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=8)

    print("Training model...")
    model = ViTForImageClassification(model_name=args.model_name, labels=num_classes)
    trainer = Trainer(gpus=args.n_gpus, max_epochs=args.epochs)
    trainer.fit(model)
    trainer.test()
    
    # save model
    print("Saving model...")
    trainer.save_checkpoint('{}/{}'.format(args.model_dir, "vit.ckpt"))