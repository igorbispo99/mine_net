import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Iterator
import os
from datetime import datetime

MINE = -1
COVERED = -2
SIZE_X = 16
SIZE_Y = 30
N_MINES = 99

class MinesweeperBoard:
    def __init__(self, height: int, width: int, n_mines: int):
        self.height = height
        self.width = width
        self.n_mines = n_mines
        self.visual_board = np.full((height, width), COVERED, dtype=np.int16)
        self.board = np.zeros((height, width), dtype=np.int16)
        self.mines_location = None
        self.non_mines_location = None
        self._initialize_board()
        
    def _initialize_board(self):
        self.board.fill(0)
        coords = np.mgrid[0:self.height, 0:self.width].reshape(2, -1).T
        
        mine_indices = np.random.choice(len(coords), self.n_mines, replace=False)
        self.mines_location = coords[mine_indices]
        
        mine_mask = np.zeros(len(coords), dtype=bool)
        mine_mask[mine_indices] = True
        self.non_mines_location = coords[~mine_mask]
        
        for y, x in self.mines_location:
            self.board[y, x] = MINE
        
        for y, x in self.mines_location:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if self.board[ny, nx] != MINE:
                            self.board[ny, nx] += 1
    
    def reset(self):
        self.visual_board.fill(COVERED)
        self._initialize_board()
    
    def get_mine_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        mask[self.mines_location[:, 0], self.mines_location[:, 1]] = 1
        return mask
    
    def generate_random_state(self, percentage_to_uncover: float) -> Tuple[np.ndarray, np.ndarray]:
        self.reset()
        
        n_to_uncover = int(len(self.non_mines_location) * percentage_to_uncover)
        indices = np.random.choice(len(self.non_mines_location), n_to_uncover, replace=False)
        coords_to_uncover = self.non_mines_location[indices]
        
        for y, x in coords_to_uncover:
            self.visual_board[y, x] = self.board[y, x]
        
        return self.visual_board.copy(), self.get_mine_mask()
    
    def _uncover_recursive(self, y: int, x: int):
        if not (0 <= y < self.height and 0 <= x < self.width):
            return
        if self.board[y, x] == MINE or self.visual_board[y, x] != COVERED:
            return
        
        self.visual_board[y, x] = self.board[y, x]
        
        if self.board[y, x] == 0:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        self._uncover_recursive(y + dy, x + dx)
    
    def click_at(self, y: int, x: int) -> bool:
        if self.board[y, x] == MINE:
            return True
        self._uncover_recursive(y, x)
        return False

class MinesweeperIterableDataset(IterableDataset):
    def __init__(self, batch_size: int = 64, clicks_max: int = 14, 
                 rand_mines: Tuple[int, int] = (40, 120)):
        self.batch_size = batch_size
        self.clicks_max = clicks_max
        self.rand_mines = rand_mines
    
    def _generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mines_n = np.random.randint(*self.rand_mines)
        board = MinesweeperBoard(SIZE_X, SIZE_Y, mines_n)
        
        if np.random.random() < 0.5:
            clicks_limit = np.random.randint(1, self.clicks_max)
            clicks = 0
            
            while clicks < clicks_limit:
                y = np.random.randint(0, board.height)
                x = np.random.randint(0, board.width)
                
                if not board.click_at(y, x):
                    clicks += 1
                else:
                    board.visual_board.fill(COVERED)
            
            visual = board.visual_board.astype(np.float32)
            mine_mask = board.get_mine_mask()
        else:
            prob = np.random.random()
            visual, mine_mask = board.generate_random_state(prob)
            visual = visual.astype(np.float32)
        
        mines_ratio = board.n_mines / (board.height * board.width)
        mines_channel = np.full((board.height, board.width), mines_ratio, dtype=np.float32)
        
        X = np.stack([visual, mines_channel], axis=0)
        Y = mine_mask[np.newaxis, ...]
        
        return torch.from_numpy(X), torch.from_numpy(Y)
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            batch_X, batch_Y = [], []
            for _ in range(self.batch_size):
                X, Y = self._generate_sample()
                batch_X.append(X)
                batch_Y.append(Y)
            
            yield torch.stack(batch_X), torch.stack(batch_Y)

class MinesweeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.output = nn.ConvTranspose2d(256, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output(x)
        return torch.sigmoid(x)

def visualize_predictions(visual_boards, true_mines, pred_mines, epoch, save_dir, n_samples=4):
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    for i in range(min(n_samples, len(visual_boards))):
        visual = visual_boards[i, 0].cpu().numpy()
        true = true_mines[i, 0].cpu().numpy()
        pred = pred_mines[i, 0].cpu().numpy()
        
        ax_visual = axes[i, 0] if n_samples > 1 else axes[0]
        ax_true = axes[i, 1] if n_samples > 1 else axes[1]
        ax_pred = axes[i, 2] if n_samples > 1 else axes[2]
        
        visual_display = visual.copy()
        visual_display[visual_display == COVERED] = -3
        im1 = ax_visual.imshow(visual_display, cmap='RdYlBu_r', vmin=-3, vmax=8)
        ax_visual.set_title(f'Board State {i+1}')
        ax_visual.grid(True, alpha=0.3)
        
        im2 = ax_true.imshow(true, cmap='Reds', vmin=0, vmax=1)
        ax_true.set_title('True Mines')
        ax_true.grid(True, alpha=0.3)
        
        im3 = ax_pred.imshow(pred, cmap='Reds', vmin=0, vmax=1)
        ax_pred.set_title(f'Predicted Mines (max: {pred.max():.3f})')
        ax_pred.grid(True, alpha=0.3)
        
        for y in range(visual.shape[0]):
            for x in range(visual.shape[1]):
                if visual[y, x] == COVERED:
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='gray', alpha=0.7)
                    ax_visual.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/predictions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics(train_losses, val_losses, epoch, save_dir):
    plt.figure(figsize=(10, 5))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{save_dir}/loss_curve_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

def train_model(model: nn.Module, steps_per_epoch: int = 1000, val_steps: int = 100,
                epochs: int = 50, device: str = 'cuda'):
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/minesweeper_{timestamp}'
    save_dir = f'checkpoints/{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/plots', exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    train_dataset = MinesweeperIterableDataset(batch_size=64)
    val_dataset = MinesweeperIterableDataset(batch_size=32)
    
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_iter = iter(train_loader)
        
        for step in range(steps_per_epoch):
            data, target = next(train_iter)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            global_step += 1
            
            if step % 100 == 0:
                writer.add_scalar('Loss/train_step', loss.item(), global_step)
                print(f'\rEpoch {epoch+1}/{epochs} - Step {step}/{steps_per_epoch} - Loss: {loss.item():.4f}', end='')
        
        avg_train_loss = train_loss / steps_per_epoch
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        val_iter = iter(val_loader)
        
        with torch.no_grad():
            for step in range(val_steps):
                data, target = next(val_iter)
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                if step == 0:
                    visualize_predictions(data, target, output, epoch+1, f'{save_dir}/plots')
                    
                    writer.add_images('Input/visual_board', data[:4, 0:1], epoch)
                    writer.add_images('Target/true_mines', target[:4], epoch)
                    writer.add_images('Prediction/pred_mines', output[:4], epoch)
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'{save_dir}/best_model.pth')
        
        if (epoch + 1) % 5 == 0:
            plot_metrics(train_losses, val_losses, epoch+1, f'{save_dir}/plots')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')
    
    writer.close()
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = MinesweeperCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trained_model = train_model(model, steps_per_epoch=1000, val_steps=100, epochs=50, device=device)
