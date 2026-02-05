import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_loss_plot(loss_curve, plot_path):
    if not loss_curve:
        return False
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return True
