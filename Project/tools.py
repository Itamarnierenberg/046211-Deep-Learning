import matplotlib.pyplot as plt
import torch
import Config as cfg


def show_examples(train_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(cfg.LABELS_MAP[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()