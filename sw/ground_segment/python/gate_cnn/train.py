import matplotlib
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import GateNet
from dataset import GateDataset

# --- Config ---
JSON_PATH = "/home/david/Downloads/MASKED_CNNv3/DATA/labels.json"
IMG_DIR = "/home/david/Downloads/MASKED_CNNv3/DATA/img"
BATCH_SIZE = 32 
EPOCHS = 50 
LEARNING_RATE = 0.001 
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2
WARMUP_EPOCHS = 5
CLASSES = ["None", "Left", "Straight", "Right"]

def calculate_class_weights(dataset):
    """Calculate weights inversely proportional to class frequency"""
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=4)
    class_counts = np.maximum(class_counts, 1) 
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 4
    return torch.tensor(weights, dtype=torch.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    full_dataset = GateDataset(JSON_PATH, IMG_DIR)
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model & Optimization
    model = GateNet(dropout_rate=DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    class_weights = calculate_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 3. Training Loop
    best_val_acc = 0
    train_losses, val_accs = [], []
    
    print(f"\nStarting Training (Weights: {class_weights.cpu().numpy()})\n" + "="*50)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, train_correct, train_total = 0, 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = 100 * val_correct / val_total
        train_losses.append(avg_train_loss)
        val_accs.append(val_accuracy)

        scheduler.step()

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "gate_model_best.pth")
        
        print(f"Ep [{epoch+1:2d}] Loss: {avg_train_loss:.3f} | Acc: {train_acc:5.1f}% | "
              f"Val Acc: {val_accuracy:5.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch > 25 and (train_acc - val_accuracy) > 25:
            print("Stopping early: Overfitting detected.")
            break

    # 4. Final Evaluation (Load Best Weights)
    print("\n" + "="*50 + "\nFINAL TEST ANALYSIS")
    model.load_state_dict(torch.load("gate_model_best.pth"))
    evaluate_model(model, test_loader, device)
    
    # 5. Visualization
    visualize_training(train_losses, val_accs)
    visualize_results(model, test_ds, device)

# --- Helper Functions ---

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Normalized Confusion Matrix (Percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Normalized Confusion Matrix (Recall per Class)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def visualize_training(train_losses, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='royalblue')
    plt.title('Training Loss Evolution')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy', color='darkorange')
    plt.title('Validation Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_results(model, dataset, device, n=5):
    model.eval()
    indices = np.random.choice(len(dataset), n)
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        out = model(img.unsqueeze(0).to(device))
        pred = out.argmax().item()
        
        plt.subplot(1, n, i+1)
        # Handle grayscale vs RGB
        disp_img = img.permute(1,2,0).cpu() if img.shape[0]==3 else img.squeeze().cpu()
        plt.imshow(disp_img, cmap='gray' if img.shape[0]==1 else None)
        
        color = 'green' if pred == label else 'red'
        plt.title(f"True: {CLASSES[label]}\nPred: {CLASSES[pred]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()