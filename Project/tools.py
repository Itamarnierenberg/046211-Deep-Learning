import matplotlib.pyplot as plt
import torch
import Config as cfg


def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range - approximately... image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1, 2, 0)


def show_examples(train_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(cfg.LABELS_MAP[label])
        plt.axis("off")
        if not cfg.RGB:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(convert_to_imshow_format(img))
    plt.show()
