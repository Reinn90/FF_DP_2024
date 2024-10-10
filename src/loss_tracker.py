import matplotlib.pyplot as plt

class LossTracker:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.layer_losses = {f'layer_{i}': [] for i in range(num_layers)}
        self.epochs = []
        
    def update(self, scalar_outputs, epoch):
        for i in range(self.num_layers):
            layer_loss = scalar_outputs[f'loss_layer_{i}']
            self.layer_losses[f'layer_{i}'].append(layer_loss)
        self.epochs.append(epoch)
    
    def plot(self, save_path='./images/layer_losses.png'):
        plt.figure(figsize=(10, 6))
        for layer, losses in self.layer_losses.items():
            plt.plot(self.epochs, losses, label=f'Layer {layer}')
            
        plt.title('Layer-wise Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
