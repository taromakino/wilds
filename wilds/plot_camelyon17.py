from argparse import ArgumentParser
from enum import Enum
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class Split(Enum):
    TRAIN = 'train'
    VAL_ID = 'id_val'
    VAL_OOD = 'val'
    TEST = 'test'


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def plot_image(ax, x):
    x = x.detach().cpu().numpy()
    if len(x.shape) == 3:
        x = x.transpose((1, 2, 0))
    ax.imshow(x, cmap='gray')


def main(data_dpath, batch_size):
    dataset = get_dataset(dataset="camelyon17", download=False, root_dir=data_dpath)
    for split in list(Split):
        test_data = dataset.get_subset(
            split.value,
            transform=transforms.ToTensor(),
        )

        n_groups = 3 if split in (Split.TRAIN, Split.VAL_ID) else 1
        grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=["hospital"])
        train_loader = get_train_loader("group", test_data, uniform_over_groups=True, grouper=grouper, n_groups_per_batch=n_groups,
            batch_size=batch_size * n_groups)
        batch = next(iter(train_loader))
        x, y, metadata = batch

        fig, axes = plt.subplots(n_groups, batch_size, figsize=(batch_size, n_groups))
        if split in (Split.TRAIN, Split.VAL_ID):
            x0, x1, x2 = x.chunk(n_groups, dim=0)
            for i in range(batch_size):
                plot_image(axes[0, i], x0[i])
                plot_image(axes[1, i], x1[i])
                plot_image(axes[2, i], x2[i])
        else:
            fig, axes = plt.subplots(n_groups, batch_size, figsize=(batch_size, n_groups))
            for i in range(batch_size):
                plot_image(axes[i], x[i])
        for ax in axes.flatten():
            remove_ticks(ax)
        plt.savefig(f'split={split.value}.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dpath', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args.data_dpath, args.batch_size)