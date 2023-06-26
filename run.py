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

    edges = []
    with open(dataset_path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            u, v = int(tokens[0]), int(tokens[1])
            if v > u:
                temp = v
                v = u
                u = temp
            edges.append([u, v])
    edges = torch.as_tensor(edges, dtype=torch.int, device=torch.device("cpu")).T

    # Run the model
    ldm = LDM(
        edges=edges, dim=dim, lr=lr, epoch_num=epoch_num, batch_size=batch_size, spe=steps_per_epoch,
        device=torch.device(device), verbose=verbose, seed=seed
    )
    # Learn the embeddings
    ldm.learn()
    # Save the model
    ldm.save(path=model_path)

    # ####################################################################################################################
    # test_intervals_num = 8
    #
    # # Load sample files
    # with open(samples_path, 'rb') as f:
    #     samples_data = pkl.load(f)
    #     labels = samples_data["test_labels"]
    #     samples = samples_data["test_samples"]
    #
    # file = open(output_path, 'w')
    #
    # # Predicted scores
    # pred_scores = [[] for _ in range(test_intervals_num)]
    # for b in range(test_intervals_num):
    #     for sample in samples[b]:
    #         i, j, t_init, t_last = sample
    #         score = ldm.get_log_intensity_sum(edges=torch.as_tensor([[i], [j]])).detach().numpy()
    #         pred_scores[b].append(score)
    #
    # for b in range(test_intervals_num):
    #
    #     if sum(labels[b]) != len(labels[b]) and sum(labels[b]) != 0:
    #         roc_auc = roc_auc_score(y_true=labels[b], y_score=pred_scores[b])
    #         file.write(f"Roc AUC, Bin Id {b}: {roc_auc}\n")
    #         pr_auc = average_precision_score(y_true=labels[b], y_score=pred_scores[b])
    #         file.write(f"PR AUC, Bin Id {b}: {pr_auc}\n")
    #         file.write("\n")
    #
    # roc_auc_complete = roc_auc_score(
    #     y_true=[l for bin_labels in labels for l in bin_labels],
    #     y_score=[s for bin_scores in pred_scores for s in bin_scores]
    # )
    # file.write(f"Roc AUC in total: {roc_auc_complete}\n")
    # pr_auc_complete = average_precision_score(
    #     y_true=[l for bin_labels in labels for l in bin_labels],
    #     y_score=[s for bin_scores in pred_scores for s in bin_scores]
    # )
    # file.write(f"PR AUC in total: {pr_auc_complete}\n")
    #
    # file.close()


if __name__ == "__main__":
    args = parse_arguments()

    process(args)