def validate_epoch(val_loader, model, criterion, device, is_topk):
    """Run validation for one epoch"""
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for input, label in val_loader:
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
            input = input[:, :torch.max(seq_len), :]
            input, label = input.float().to(device), label.float().to(device)

            logits, logits2 = model(input, seq_len)
            clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
            clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
            croloss = CENTROPY(logits, logits2, seq_len, device)

            total_loss += (clsloss + clsloss2 + 5*croloss).item()
            batch_count += 1

    return total_loss / batch_count

def test_hlnet_single(dataloader, model, device):
    """Test function that evaluates video-level predictions"""
    print("Starting test...")
    with torch.no_grad():
        model.eval()
        pred = []
        pred2 = []
        gt = []
        
        for i, (input, label) in enumerate(dataloader):
            gt.append(label[0].item())  # Take first label as they're all same for a video
            
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            
            # Get predictions for offline model
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            batch_pred = torch.mean(sig).item()
            pred.append(batch_pred)
            
            # Get predictions for online model
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            batch_pred2 = torch.mean(sig2).item()
            pred2.append(batch_pred2)

        # Convert to numpy arrays
        gt = np.array(gt)
        pred = np.array(pred)
        pred2 = np.array(pred2)

        # Calculate metrics
        precision, recall, _ = precision_recall_curve(gt, pred)
        pr_auc = auc(recall, precision)
        
        precision2, recall2, _ = precision_recall_curve(gt, pred2)
        pr_auc2 = auc(recall2, precision2)
        
        return pr_auc, pr_auc2

def train_hlnet_single(train_loader, model, optimizer, scheduler, criterion,
                      device, is_topk, val_loader=None):
    """Training function with loss tracking"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for i, (input, label) in enumerate(train_loader):
        seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
        input = input[:, :torch.max(seq_len), :]
        input, label = input.float().to(device), label.float().to(device)

        logits, logits2 = model(input, seq_len)
        clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
        clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
        croloss = CENTROPY(logits, logits2, seq_len, device)

        total_loss = clsloss + clsloss2 + 5*croloss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        batch_count += 1

        if (i + 1) % 100 == 0:
            print(f"Step {i+1}: Training Loss: {total_loss.item():.4f}")

    avg_epoch_loss = epoch_loss / batch_count

    # Calculate validation loss if provided
    val_epoch_loss = None
    if val_loader is not None:
        val_epoch_loss = validate_epoch(val_loader, model, criterion, device, is_topk)

    return model, avg_epoch_loss, val_epoch_loss

def plot_training_history(train_losses, val_losses, modality, save_path=None):
    """Plot training and validation losses"""
    if save_path is None:
        save_path = f'hlnet_{modality.lower()}_training_history.png'
        
    plt.figure(figsize=(12, 8))
    epochs = range(len(train_losses))

    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')

    plt.title(f'HL-Net {modality} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Initialize settings
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set feature size based on modality
    if args.modality == 'RGB':
        args.feature_size = 1024
    elif args.modality == 'AUDIO':
        args.feature_size = 128
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")
    
    # Create model and move to device
    model = Model(args).to(device)

    # Setup optimizer
    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    optimizer = optim.Adam([
        {'params': base_param},
        {'params': model.approximator.parameters(), 'lr': args.lr / 2},
        {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
    ], lr=args.lr, weight_decay=0.000)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    criterion = torch.nn.BCELoss()
    is_topk = True

    # Create data loaders using your existing function
    train_loader, val_loader, test_loader = create_single_modality_data_loaders(args)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Training tracking
    train_losses = []
    val_losses = []
    best_pr_auc = 0
    best_epoch = 0

    print(f"Starting training on {device} for {args.modality} modality")

    for epoch in range(args.max_epoch):
        print(f"\nEpoch {epoch+1}/{args.max_epoch}")
        st = time.time()

        # Training step
        model, train_loss, val_loss = train_hlnet_single(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            is_topk=is_topk,
            val_loader=val_loader
        )

        # Store losses
        train_losses.append(train_loss)
        if val_loss is not None:
            val_losses.append(val_loss)

        # Calculate PR-AUC for monitoring
        pr_auc, pr_auc_online = test_hlnet_single(test_loader, model, device)

        # Update best PR-AUC
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            # Save best model
            save_dir = f'./hlnet_saves_{args.modality.lower()}/best'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/{args.model_name}_best.pth')

        print(f'Epoch {epoch+1}/{args.max_epoch}:')
        print(f'Train Loss: {train_loss:.4f}')
        if val_loss is not None:
            print(f'Validation Loss: {val_loss:.4f}')
        print(f'PR-AUC (Offline/Online): {pr_auc:.4f}/{pr_auc_online:.4f}')
        print(f'Best PR-AUC: {best_pr_auc:.4f} (Epoch {best_epoch+1})')
        print(f'Epoch time: {time.time() - st:.2f}s')

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.get_best_model_state())
            break

        scheduler.step()

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 and epoch > 0:
            save_dir = f'./hlnet_saves_{args.modality.lower()}/checkpoints'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), 
                      f'{save_dir}/{args.model_name}_epoch{epoch}.pth')

    # Save final model
    save_dir = f'./hlnet_saves_{args.modality.lower()}/final'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), 
              f'{save_dir}/{args.model_name}_final.pth')

    # Plot and save training history
    plot_training_history(train_losses, val_losses, args.modality)

    # Final evaluation
    final_pr_auc, final_pr_auc_online = test_hlnet_single(test_loader, model, device)
    print("\nTraining completed!")
    print(f'Final PR-AUC (Offline/Online): {final_pr_auc:.4f}/{final_pr_auc_online:.4f}')
    print(f'Best PR-AUC: {best_pr_auc:.4f} (Epoch {best_epoch+1})')

if __name__ == '__main__':
    main()
