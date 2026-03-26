import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from model import GateNet
from dataset import GateDataset

# Configuration - OPTIMIZED
JSON_PATH = "./DATA/labels.json"
IMG_DIR = "./DATA/img"
BATCH_SIZE = 32  # Reduced for better learning
EPOCHS = 50  # More epochs to see if it improves
LEARNING_RATE = 0.001  # Higher learning rate
WEIGHT_DECAY = 0  # No L2 regularization initially
DROPOUT_RATE = 0.2  # Lower dropout
WARMUP_EPOCHS = 5  # Warmup period

def calculate_class_weights(dataset):
    """Calculate weights inversely proportional to class frequency"""
    labels = [item[1].item() for item in dataset]
    class_counts = np.bincount(labels, minlength=4)
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 4
    return torch.tensor(weights, dtype=torch.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Loading dataset...")
    full_dataset = GateDataset(JSON_PATH, IMG_DIR)
    print(f"Total samples: {len(full_dataset)}")
    
    # Check class distribution
    labels = [item[1].item() for item in full_dataset]
    class_counts = np.bincount(labels, minlength=4)
    print(f"Class distribution: {class_counts}")
    print(f"  Class 0 (None): {class_counts[0]}")
    print(f"  Class 1 (Left): {class_counts[1]}")
    print(f"  Class 2 (Straight): {class_counts[2]}")
    print(f"  Class 3 (Right): {class_counts[3]}")
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Initialize Model
    print("\nInitializing model...")
    model = GateNet(dropout_rate=DROPOUT_RATE).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    # No scheduler for now - let's see raw behavior
    class_weights = calculate_class_weights(train_ds)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss() #(weight=class_weights)

    # 3. Training Loop with debugging
    best_val_acc = 0
    train_losses = []
    val_accs = []
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            #optimizer.zero_grad()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 		T_max=EPOCHS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_accs.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "gate_model_best.pth")
        
        # Print progress
        gap = train_acc - val_accuracy
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:5.1f}% | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:5.1f}% | Gap: {gap:5.1f}%")
        
        # Early stopping if overfitting badly
        if epoch > 20 and gap > 20:
            print(f"⚠️ Overfitting detected (gap={gap:.1f}%). Stopping early.")
            break
        scheduler.step()
	
    # 4. Evaluate on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    evaluate_model(model, test_loader, device)

    # 5. Save final model
    torch.save(model.state_dict(), "gate_model_final.pth")
    
    # 6. Visualize
    visualize_training(train_losses, val_accs)
    visualize_results(model, val_ds, device, n=8)
    
    # 7. Per-class analysis
    analyze_per_class(model, test_loader, device)

def evaluate_model(model, dataloader, device):
    """Evaluate model on test set"""
    classes = ["None", "Left", "Straight", "Right"]
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels) * 100
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    
    # Per-class metrics
    print(f"\n{'Class':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 40)
    
    for class_idx, class_name in enumerate(classes):
        mask = all_labels == class_idx
        if mask.sum() > 0:
            class_acc = np.mean(all_preds[mask] == all_labels[mask]) * 100
            support = mask.sum()
            print(f"{class_name:<12} {class_acc:6.2f}% {int(support):<10}")
    
    # Confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        cm[true_label, pred_label] += 1
    
    print(f"\nConfusion Matrix:")
    print("         " + "  ".join(f"{c:>8}" for c in classes))
    for i, row in enumerate(cm):
        print(f"{classes[i]:<8} {row}")

def analyze_per_class(model, dataloader, device):
    """Detailed per-class analysis"""
    classes = ["None", "Left", "Straight", "Right"]
    model.eval()
    
    class_probs = {i: [] for i in range(4)}
    class_correct = {i: [] for i in range(4)}
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            for i in range(4):
                mask = labels == i
                if mask.sum() > 0:
                    class_probs[i].extend(probs[mask, i].cpu().numpy())
                    class_correct[i].extend((preds[mask] == labels[mask]).cpu().numpy())
    
    print("\nPer-Class Confidence Analysis:")
    print(f"{'Class':<12} {'Avg Confidence':<18} {'Accuracy':<12}")
    print("-" * 45)
    for i, class_name in enumerate(classes):
        if len(class_probs[i]) > 0:
            avg_conf = np.mean(class_probs[i]) * 100
            accuracy = np.mean(class_correct[i]) * 100
            print(f"{class_name:<12} {avg_conf:6.2f}% {accuracy:6.2f}%")

def visualize_training(train_losses, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_accs, label='Validation Accuracy', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("\nSaved training_history.png")
    plt.show()

def visualize_results(model, dataset, device, n=8):
    """Visualize predictions"""
    classes = ["None", "Left", "Straight", "Right"]
    model.eval()
    
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    
    plt.figure(figsize=(16, 3))
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax().item()
            confidence = torch.softmax(output, dim=1)[0, pred].item()
        
        plt.subplot(1, n, i+1)
        plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
        color = 'green' if pred == label else 'red'
        plt.title(f"True: {classes[label]}\nPred: {classes[pred]}\nConf: {confidence:.2f}", 
                 color=color, fontweight='bold', fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=100)
    print("Saved sample_predictions.png")
    plt.show()

if __name__ == "__main__":
    main()
