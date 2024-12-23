import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNIST_CNN
import datetime
import os
import logging
from tqdm import tqdm

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup logging configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def train_model():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        logger.info("Created models directory")
    
    # Force CPU usage if running on GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        device = torch.device('cpu')
        logger.info("Running on GitHub Actions - forcing CPU usage")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Log the number of samples in the datasets
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of testing samples: {len(test_dataset)}")
    
    # Adjust workers based on environment
    num_workers = 4 #0 if os.getenv('GITHUB_ACTIONS') else 2
    
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,  # Use same batch size as training
        shuffle=False,
        num_workers=num_workers
    )
    logger.info(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Initialize model, optimizer, and loss function
    model = MNIST_CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    #criterion = nn.CrossEntropyLoss()
    
    param_count = count_parameters(model)
    logger.info(f"Model initialized with {param_count} parameters")
    logger.info(f"Model architecture:\n{model}")
    
    # Training loop
    best_accuracy = 0
    best_model_path = None
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/20')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = output, target
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            pred = output.argmax(dim=1)
            correct_train += pred.eq(target).sum().item()
            total_train += target.size(0)
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'train_acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        # Calculate epoch training statistics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        logger.info("Starting evaluation...")
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Evaluating')
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += (output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Update test progress bar
                test_pbar.set_postfix({
                    'test_acc': f'{100.*correct/total:.2f}%'
                })
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        
        # Log epoch statistics
        logger.info(f'Epoch {epoch+1}/20:')
        logger.info(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model with timestamp in models directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join('models', f'model_acc{accuracy:.2f}_params{param_count}_{timestamp}.pth')
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            logger.info(f'New best model saved: {model_path}')
    
    # Enhanced final summary
    logger.info("="*50)
    logger.info("Training Summary:")
    logger.info(f"Total Parameters: {param_count:,}")
    logger.info(f"Best Test Accuracy: {best_accuracy:.2f}%")
    if best_model_path:
        logger.info(f"Best Model Path: {best_model_path}")
        logger.info(f"Model Size: {os.path.getsize(best_model_path)/1024:.2f} KB")
    logger.info("="*50)
    
    return best_accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    train_model() 