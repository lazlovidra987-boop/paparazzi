import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
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

def calculate_class_weights(dataset):
    """Calculate weights inversely proportional to class frequency"""
    # Extract labels from the subset/dataset
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=4)
    class_counts = np.maximum(class_counts, 1) # Avoid division by zero
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
    
    # Optimizer initialized ONCE
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Class weights for imbalanced data
    class_weights = calculate_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Learning Rate Schedulers
    # 1. Linear warmup for the first few epochs
    # 2. Cosine annealing for the rest
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
            
            # --- Gradient Step ---
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # --- Metrics ---
            total_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accs.append(val_accuracy)

        # Update LR Scheduler
        scheduler.step()

        # Save Best
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "gate_model_best.pth")
        
        gap = train_acc - val_accuracy
        print(f"Ep [{epoch+1:2d}] Loss: {avg_train_loss:.3f} | Acc: {train_acc:5.1f}% | "
              f"Val Acc: {val_accuracy:5.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch > 25 and gap > 25:
            print("Stopping early: Overfitting detected.")
            break

    # 4. Final Evaluation
    print("\n" + "="*50 + "\nFINAL TEST RESULTS")
    evaluate_model(model, test_loader, device)
    
    # 5. Visualization
    visualize_training(train_losses, val_accs)
    visualize_results(model, val_ds, device)

# --- Helper Functions (From your original script) ---

def evaluate_model(model, dataloader, device):
    classes = ["None", "Left", "Straight", "Right"]
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"Overall Test Accuracy: {accuracy:.2f}%")

def visualize_training(train_losses, val_accs):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss')
    plt.title('Train Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, color='orange', label='Val Acc')
    plt.title('Validation Accuracy')
    plt.show()

def visualize_results(model, dataset, device, n=5):
    classes = ["None", "Left", "Straight", "Right"]
    model.eval()
    indices = np.random.choice(len(dataset), n)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        out = model(img.unsqueeze(0).to(device))
        pred = out.argmax().item()
        plt.subplot(1, n, i+1)
        plt.imshow(img.permute(1,2,0).cpu() if img.shape[0]==3 else img.squeeze().cpu(), cmap='gray')
        plt.title(f"T: {classes[label]}\nP: {classes[pred]}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()