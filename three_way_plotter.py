import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_decision_boundary_3w(model: torch.nn.Module, X: torch.tensor, y: torch.tensor):
    model.eval()
    with torch.inference_mode():
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                             np.linspace(y_min, y_max, 101))
        grid = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

        preds = model(grid)
        preds = torch.softmax(preds, dim=1)
        class_preds = torch.argmax(preds, dim=1).reshape(xx.shape)

        plt.contourf(xx, yy, class_preds, cmap=plt.cm.RdYlBu, alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=40)