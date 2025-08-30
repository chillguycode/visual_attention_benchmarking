# run_experiment.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import os

from src.data_loader import get_dataset
from src.model import CNNWithAttention, BasicBlock
from src.utils import cleanup

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    loop = tqdm(loader, desc="Training")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_accuracy = 100 * correct / total
    return total_loss / len(loader), train_accuracy

def evaluate_final(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(loader, desc="Final Testing")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def run(config):
    """Main function to run a single experiment configuration."""
    
    # Set the MLflow experiment name
    mlflow.set_experiment(config.experiment_name)
    
    # Start an MLflow run to log everything for this specific experiment
    with mlflow.start_run():
        print(f"INFO: Starting run for attention: {config.attention_type} on dataset: {config.dataset}")
        
        # Log all our configuration parameters to MLflow for tracking
        mlflow.log_params(vars(config))

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load data using our modular data loader
        train_ds, test_ds, num_classes, in_channels = get_dataset(config.dataset)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # Build the model using our modular model definition
        model = CNNWithAttention(
            BasicBlock, [2, 2, 2, 2], 
            num_classes=num_classes, 
            in_channels=in_channels, 
            attention_type=config.attention_type
        ).to(DEVICE)
        
        # Standard setup
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

        # Training loop
        for epoch in range(config.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler)
            scheduler.step()
            
            # Log metrics to MLflow after each epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            
            print(f"Epoch {epoch+1}/{config.epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Final evaluation
        final_accuracy = evaluate_final(model, test_loader, DEVICE)
        print(f"\nFINAL TEST ACCURACY: {final_accuracy:.2f}%\n")
        
        # Log the final, most important metric
        mlflow.log_metric("final_test_accuracy", final_accuracy)
        
        # Save the model artifact
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/{config.dataset}_{config.attention_type}_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Clean up GPU memory before the next run
        cleanup(model, optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CNN Attention Mechanism Experiments")
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'Oxford102Flowers', 'EuroSAT'], help='Dataset to use for the experiment.')
    parser.add_argument('--attention_type', type=str, default='none', choices=['none', 'se', 'cbam'], help='Type of attention mechanism to use.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--experiment_name', type=str, default="Visual Attention Benchmark", help="Name for the MLflow experiment.")
    
    args = parser.parse_args()
    run(args)
