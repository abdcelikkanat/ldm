import torch
from torch_sparse import spspmm
import math
import time
import sys
import random


class LDM(torch.nn.Module):
    def __init__(self,  edges, weights=None, dim=2, lr=0.1, epoch_num=100, batch_size = 0, spe=1,
                 device=torch.device("cpu"), verbose=False, seed=0):

        super(LDM, self).__init__()

        self.__edges = edges.to(device)
        self.__weights = torch.ones((self.__edges.shape[1],), dtype=torch.float, device=device) if weights is None else weights.to(device)
        self.__nodes_num = torch.max(self.__edges) + 1
        self.__edges_num = self.__edges.shape[1]
        self.__dim = dim
        self.__sampling_weights = torch.ones(self.__nodes_num, dtype=torch.float, device=device)
        self.__device = device
        self.__seed = seed
        self.__verbose = verbose

        self.__beta = torch.nn.Parameter(
            2 * torch.rand(size=(self.__nodes_num,), device=self.__device) - 1, requires_grad=True
        )
        self.__z = torch.nn.Parameter(
            2 * torch.rand(size=(self.__nodes_num, self.__dim), device=self.__device) - 1, requires_grad=True
        )

        self.__epoch_num = epoch_num
        self.__steps_per_epoch = spe
        self.__batch_size = batch_size if batch_size else self.__nodes_num
        self.__learning_rate = lr
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__learning_rate)
        self.__loss = []

        self.__pdist = torch.nn.PairwiseDistance(p=2)

    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def get_intensity_sum(self, nodes=None):

        beta = self.__beta if nodes is None else torch.index_select(self.__beta, index=nodes, dim=0)
        z = self.__z if nodes is None else torch.index_select(self.__z, index=nodes, dim=0)

        beta_mat = beta.unsqueeze(0) + beta.unsqueeze(1)
        dist_mat = torch.cdist(z, z, p=2)

        return torch.triu(torch.exp(beta_mat - dist_mat), diagonal=1).sum()

    def get_log_intensity_sum(self, edges, weights):

        beta_pair = torch.index_select(self.__beta, index=edges[0], dim=0) + \
                    torch.index_select(self.__beta, index=edges[1], dim=0)

        z_dist = self.__pdist(
            torch.index_select(self.__z, index=edges[0], dim=0),
            torch.index_select(self.__z, index=edges[1], dim=0),
        )

        return (weights * (beta_pair - z_dist)).sum()

    def get_intensity_for(self, i, j):

        beta_sum = self.__beta[i] + self.__beta[j]
        z_dist = self.__pdist(self.__z[i, :], self.__z[j, :])
        return torch.exp(beta_sum - z_dist)

    def get_neg_likelihood(self, edges, weights, nodes=None):

        # Compute the link term
        link_term = self.get_log_intensity_sum(edges=edges, weights=weights)

        # Compute the non-link term
        non_link = self.get_intensity_sum(nodes=nodes)

        return -(link_term - non_link)

    def learn(self):

        for epoch in range(self.__epoch_num):

            self.__train_one_epoch(current_epoch=epoch)

        return self.__loss

    def __train_one_epoch(self, current_epoch):

        init_time = time.time()

        total_batch_loss = 0
        self.__loss.append([])
        for batch_num in range(self.__steps_per_epoch):
            batch_loss = self.__train_one_batch()

            self.__loss[-1].append(batch_loss)

            total_batch_loss += batch_loss

            # Set the gradients to 0
            self.__optimizer.zero_grad()

            # Backward pass
            batch_loss.backward()

            # Perform a step
            self.__optimizer.step()

        # Get the average epoch loss
        epoch_loss = total_batch_loss / float(self.__steps_per_epoch)

        if not math.isfinite(epoch_loss):
            print(f"Epoch loss is {epoch_loss}, stopping training")
            sys.exit(1)

        if self.__verbose and (current_epoch % 10 == 0 or current_epoch == self.__epoch_num - 1):
            print(f"| Epoch = {current_epoch} | Loss/train: {epoch_loss} | Epoch Elapsed time: {time.time() - init_time}")

    def __train_one_batch(self):

        self.train()

        sampled_nodes = torch.multinomial(self.__sampling_weights, self.__batch_size, replacement=False)
        sampled_nodes, _ = torch.sort(sampled_nodes, dim=0)

        batch_edges, batch_weights = spspmm(
            indexA=self.__edges.type(torch.long),
            valueA=self.__weights,
            indexB=torch.vstack((sampled_nodes, sampled_nodes)).type(torch.long),
            valueB=torch.ones(size=(self.__batch_size,), dtype=torch.float, device=self.__device),
            m=self.__nodes_num, k=self.__nodes_num, n=self.__nodes_num, coalesced=True
        )

        # Forward pass
        average_batch_loss = self.forward(edges=batch_edges, weights=batch_weights, nodes=sampled_nodes)

        return average_batch_loss

    def forward(self, edges, weights, nodes):

        nll = self.get_neg_likelihood(edges=edges, weights=weights, nodes=nodes)

        return nll

    def get_params(self):

        return self.__beta.detach().numpy(), self.__z.detach().numpy()

    def save(self, path):

        if self.__verbose:
            print(f"- Model file is saving.")
            print(f"\t+ Target path: {path}")

        kwargs = {
            'edges': self.__edges,
            'dim': self.__dim,
            'lr': self.__learning_rate,
            'epoch_num': self.__epoch_num,
            'batch_size': self.__batch_size,
            'spe': self.__steps_per_epoch,
            'device': self.__device,
            'verbose': self.__verbose,
            'seed': self.__seed
        }

        torch.save([kwargs, self.state_dict()], path)

        if self.__verbose:
            print(f"\t+ Completed.")

    def save_embs(self, path, format="word2vec"):

        assert format == "word2vec", "Only acceptable format is word2vec."

        with open(path, 'w') as f:
            f.write(f"{self.__nodes_num} {self.__dim}\n")
            for i in range(self.__nodes_num):
                f.write("{} {}\n".format(i, ' '.join(str(value) for value in self.__z[i, :])))
