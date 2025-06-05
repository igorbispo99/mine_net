import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cv2
from typing import List, Tuple, Dict
from mine_net_v2 import *
MINE = -1
COVERED = -2
SIZE_X = 16
SIZE_Y = 30
N_MINES = 99
class ModelAnalyzer:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.eval()
        self.device = device
        self.activations = {}
        self.gradients = {}
        
    def register_hooks(self):
        """Registra hooks para capturar ativações e gradientes"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Registrar hooks nas camadas principais
        self.model.block1.register_forward_hook(forward_hook('block1'))
        self.model.block2.register_forward_hook(forward_hook('block2'))
        self.model.block3.register_forward_hook(forward_hook('block3'))
        self.model.block4.register_forward_hook(forward_hook('block4'))
        
        self.model.block4.register_backward_hook(backward_hook('block4'))
    
    def visualize_filters(self, layer_name: str = 'block1.0', max_filters: int = 32):
        """Visualiza os filtros da primeira camada convolucional"""
        # Acessar primeira conv do bloco
        if layer_name == 'block1.0':
            conv_layer = self.model.block1[0]
        elif layer_name == 'block2.0':
            conv_layer = self.model.block2[0]
        else:
            raise ValueError(f"Layer {layer_name} not supported")
        
        filters = conv_layer.weight.data.cpu().numpy()
        n_filters = min(filters.shape[0], max_filters)
        
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(n_filters):
            # Média dos canais de entrada para visualização
            filter_vis = filters[i].mean(axis=0)
            
            im = axes[i].imshow(filter_vis, cmap='RdBu_r')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        plt.suptitle(f'Convolutional Filters - {layer_name}')
        plt.tight_layout()
        plt.show()
    
    def get_activation_maps(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Obtém mapas de ativação para uma entrada"""
        self.register_hooks()
        self.activations.clear()
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return self.activations.copy()
    
    def visualize_activations(self, input_tensor: torch.Tensor, sample_idx: int = 0):
        """Visualiza mapas de ativação de diferentes camadas"""
        activations = self.get_activation_maps(input_tensor)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Entrada original
        visual_board = input_tensor[sample_idx, 0].cpu().numpy()
        axes[0, 0].imshow(visual_board, cmap='RdYlBu_r', vmin=-2, vmax=8)
        axes[0, 0].set_title('Input (Visual Board)')
        
        # Saída do modelo
        output = self.model(input_tensor)
        pred_mines = output[sample_idx, 0].detach().cpu().numpy()
        axes[0, 1].imshow(pred_mines, cmap='Reds', vmin=0, vmax=1)
        axes[0, 1].set_title('Model Output')
        
        # Ativações médias de cada bloco
        block_names = ['block1', 'block2', 'block3', 'block4']
        positions = [(0, 2), (0, 3), (1, 0), (1, 1)]
        
        for block_name, pos in zip(block_names, positions):
            if block_name in activations:
                # Média dos canais
                act = activations[block_name][sample_idx].mean(dim=0).cpu().numpy()
                im = axes[pos].imshow(act, cmap='viridis')
                axes[pos].set_title(f'{block_name} (avg activation)')
                plt.colorbar(im, ax=axes[pos], fraction=0.046)
        
        # Remover eixos vazios
        for i in range(2, 4):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def grad_cam(self, input_tensor: torch.Tensor, target_layer: str = 'block4', 
                 sample_idx: int = 0, target_position: Tuple[int, int] = None):
        """Implementa Grad-CAM para visualizar regiões importantes"""
        self.register_hooks()
        self.activations.clear()
        self.gradients.clear()
        
        input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Se não especificar posição, usar a de maior probabilidade
        if target_position is None:
            pred = output[sample_idx, 0]
            target_position = np.unravel_index(pred.argmax().cpu(), pred.shape)
        
        # Backward pass para a posição específica
        self.model.zero_grad()
        output[sample_idx, 0, target_position[0], target_position[1]].backward()
        
        # Calcular Grad-CAM
        gradients = self.gradients[target_layer][sample_idx]
        activations = self.activations[target_layer][sample_idx]
        
        # Pesos médios
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Combinação linear ponderada
        grad_cam = (weights * activations).sum(dim=0)
        grad_cam = F.relu(grad_cam)
        
        # Redimensionar para tamanho original
        grad_cam = grad_cam.unsqueeze(0).unsqueeze(0)
        grad_cam = F.interpolate(grad_cam, size=(16, 30), mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze().cpu().numpy()
        
        # Normalizar
        if grad_cam.max() > 0:
            grad_cam = grad_cam / grad_cam.max()
        
        return grad_cam
    
    def analyze_patterns(self, n_samples: int = 1000):
        """Analisa padrões comuns que o modelo identifica"""
        patterns = {
            'isolated_covered': [],
            'numbered_boundary': [],
            'large_covered_area': [],
            'corner_cells': []
        }
        
        dataset = MinesweeperIterableDataset(batch_size=1)
        data_iter = iter(DataLoader(dataset, batch_size=None))
        
        for _ in range(n_samples):
            input_tensor, target = next(data_iter)
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            
            visual = input_tensor[0, 0].cpu().numpy()
            pred = output[0, 0].cpu().numpy()
            
            # Analisar diferentes padrões
            for i in range(16):
                for j in range(30):
                    if visual[i, j] == COVERED:
                        # Célula coberta isolada
                        neighbors_covered = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < 16 and 0 <= nj < 30:
                                    if visual[ni, nj] == COVERED:
                                        neighbors_covered += 1
                        
                        if neighbors_covered == 0:
                            patterns['isolated_covered'].append(pred[i, j])
                        
                        # Células nos cantos
                        if (i == 0 or i == 15) and (j == 0 or j == 29):
                            patterns['corner_cells'].append(pred[i, j])
        
        # Estatísticas dos padrões
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (pattern_name, values) in enumerate(patterns.items()):
            if values:
                axes[idx].hist(values, bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{pattern_name}\nMean: {np.mean(values):.3f}')
                axes[idx].set_xlabel('Predicted Probability')
                axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def occlusion_analysis(self, input_tensor: torch.Tensor, window_size: int = 3,
                          stride: int = 1, sample_idx: int = 0):
        """Análise de oclusão para entender importância de diferentes regiões"""
        original_input = input_tensor.clone()
        
        with torch.no_grad():
            original_output = self.model(original_input)[sample_idx, 0]
        
        importance_map = np.zeros((16, 30))
        
        for i in range(0, 16 - window_size + 1, stride):
            for j in range(0, 30 - window_size + 1, stride):
                # Criar cópia e ocluir região
                occluded_input = original_input.clone()
                occluded_input[sample_idx, 0, i:i+window_size, j:j+window_size] = COVERED
                
                with torch.no_grad():
                    occluded_output = self.model(occluded_input)[sample_idx, 0]
                
                # Diferença média na região
                diff = torch.abs(original_output - occluded_output).mean().item()
                importance_map[i:i+window_size, j:j+window_size] += diff
        
        # Normalizar pelo número de sobreposições
        importance_map = importance_map / (window_size * window_size)
        
        return importance_map
    
    def visualize_decision_process(self, input_tensor: torch.Tensor, target: torch.Tensor,
                                 sample_idx: int = 0):
        """Visualização completa do processo de decisão do modelo"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Entrada e saída
        ax1 = plt.subplot(3, 4, 1)
        visual = input_tensor[sample_idx, 0].cpu().numpy()
        im1 = ax1.imshow(visual, cmap='RdYlBu_r', vmin=-2, vmax=8)
        ax1.set_title('Input Board State')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = plt.subplot(3, 4, 2)
        true_mines = target[sample_idx, 0].cpu().numpy()
        ax2.imshow(true_mines, cmap='Reds', vmin=0, vmax=1)
        ax2.set_title('True Mines')
        
        ax3 = plt.subplot(3, 4, 3)
        with torch.no_grad():
            pred_mines = self.model(input_tensor)[sample_idx, 0].cpu().numpy()
        im3 = ax3.imshow(pred_mines, cmap='Reds', vmin=0, vmax=1)
        ax3.set_title('Predicted Mines')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 2. Grad-CAM para diferentes posições
        high_prob_pos = np.unravel_index(pred_mines.argmax(), pred_mines.shape)
        grad_cam_high = self.grad_cam(input_tensor, sample_idx=sample_idx, 
                                     target_position=high_prob_pos)
        
        ax4 = plt.subplot(3, 4, 5)
        im4 = ax4.imshow(grad_cam_high, cmap='hot', alpha=0.7)
        ax4.imshow(visual, cmap='gray', alpha=0.3)
        ax4.set_title(f'Grad-CAM (High Prob @ {high_prob_pos})')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # 3. Importância por oclusão
        ax5 = plt.subplot(3, 4, 6)
        importance = self.occlusion_analysis(input_tensor, sample_idx=sample_idx)
        im5 = ax5.imshow(importance, cmap='viridis')
        ax5.set_title('Occlusion Importance')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 4. Diferença entre predição e realidade
        ax6 = plt.subplot(3, 4, 7)
        diff = pred_mines - true_mines
        im6 = ax6.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        ax6.set_title('Prediction Error')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # 5. Histograma de probabilidades
        ax7 = plt.subplot(3, 4, 9)
        ax7.hist(pred_mines.flatten(), bins=50, alpha=0.7)
        ax7.set_xlabel('Predicted Probability')
        ax7.set_ylabel('Count')
        ax7.set_title('Probability Distribution')
        
        # 6. Análise por tipo de célula
        ax8 = plt.subplot(3, 4, 10)
        covered_probs = pred_mines[visual == COVERED]
        uncovered_probs = []
        for val in range(9):
            mask = visual == val
            if mask.any():
                uncovered_probs.extend(pred_mines[mask])
        
        ax8.boxplot([covered_probs, uncovered_probs], labels=['Covered', 'Uncovered'])
        ax8.set_ylabel('Predicted Probability')
        ax8.set_title('Predictions by Cell Type')
        
        plt.tight_layout()
        plt.show()

# Função para carregar e analisar modelo
def analyze_trained_model(model_path: str, device: str = 'cuda'):
    # Carregar modelo
    model = MinesweeperCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Criar analisador
    analyzer = ModelAnalyzer(model, device)
    
    # Gerar algumas amostras para análise
    dataset = MinesweeperIterableDataset(batch_size=8)
    data_loader = DataLoader(dataset, batch_size=None)
    input_tensor, target = next(iter(data_loader))
    input_tensor, target = input_tensor.to(device), target.to(device)
    
    print("1. Visualizando filtros da primeira camada...")
    analyzer.visualize_filters('block1.0')
    
    print("\n2. Visualizando ativações...")
    analyzer.visualize_activations(input_tensor, sample_idx=0)
    
    print("\n3. Analisando processo de decisão completo...")
    analyzer.visualize_decision_process(input_tensor, target, sample_idx=0)
    
    print("\n4. Analisando padrões aprendidos...")
    analyzer.analyze_patterns(n_samples=500)
    
    return analyzer

# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho do seu modelo treinado
    model_path = r"C:\Users\igorbispo\Documents\checkpoints\20250605_174056\best_model.pth"
    analyzer = analyze_trained_model(model_path)