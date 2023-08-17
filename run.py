import torch
from argparse import ArgumentParser, RawTextHelpFormatter
from src.ldm import LDM

# Global control for device
CUDA = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (CUDA) and (device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--dataset', type=str, required=True, help='Path of the dataset'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--dim', type=int, default=2, required=False, help='Dimension size'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=500, required=False, help='Number of epochs'
    )
    parser.add_argument(
        '--spe', type=int, default=10, required=False, help='Number of steps per epoch'
    )
    parser.add_argument(
        '--batch_size', type=int, default=100, required=False, help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1, required=False, help='Learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument(
        '--verbose', type=bool, default=1, required=False, help='Verbose'
    )

    return parser.parse_args()


def process(args):

    dataset_path = args.dataset
    model_path = args.model_path
    dim = args.dim
    epoch_num = args.epoch_num
    steps_per_epoch = args.spe
    batch_size = args.batch_size
    lr = args.lr

    seed = args.seed
    verbose = args.verbose

    edges, weights = [], []
    with open(dataset_path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            u, v = int(tokens[0]), int(tokens[1])
            if v > u:
                temp = v
                v = u
                u = temp

            # Add the edge
            edges.append([u, v])
            # Add the weight
            if len(tokens) > 2:
                weights.append(float(tokens[2]))

    edges = torch.as_tensor(edges, dtype=torch.int, device=torch.device("cpu")).T
    weights = torch.as_tensor(weights, dtype=torch.float, device=torch.device("cpu"))

    # Run the model
    ldm = LDM(
        edges=edges, dim=dim, lr=lr, epoch_num=epoch_num, batch_size=batch_size, spe=steps_per_epoch,
        device=torch.device(device), verbose=verbose, seed=seed
    )
    # Learn the embeddings
    ldm.learn()
    # Save the model
    ldm.save(path=model_path)


if __name__ == "__main__":
    args = parse_arguments()

    process(args)