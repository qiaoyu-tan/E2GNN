import os.path as osp
import numpy as np
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
import torch
import time
from utils import delete_edges


def get_dataset_split(path, name, train_examples_per_class, val_examples_per_class, ind_ratio):
    datasets = get_dataset(path, name)
    data = datasets[0]
    x, edge_index, y = data.x, data.edge_index, data.y.view(-1)
    num_class = torch.max(y) + 1
    ones_hot = torch.eye(num_class)
    labels = ones_hot[y].data.numpy()
    if name == 'Cora' or name == 'CiteSeer' or name == 'PubMed':
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = torch.zeros_like(data.test_mask, dtype=torch.bool)
        idx_train = torch.nonzero(train_mask.int())
        idx_val = torch.nonzero(val_mask.int())
        idx_test_origin = torch.nonzero(data.test_mask.int()).view(-1).numpy()

        non_test = set(torch.cat([idx_train, idx_val], dim=0).view(-1).data.numpy().tolist())
        all_test = set(list(range(x.shape[0])))
        idx_test = list(all_test.difference(non_test))
        ind_len = int(len(idx_test) * ind_ratio)
        if ind_len <= idx_test_origin.shape[0]:
            random_state = np.random.RandomState(0)
            ind_test_idx = random_state.choice(idx_test_origin, ind_len, replace=False)
        else:
            ind_test_idx = idx_test_origin

        u_trains_idx = np.array(list(set(idx_test).difference(set(ind_test_idx.tolist()))))
        # build indctive edge index
        edge_index_np = edge_index.numpy()
        edge_index_train = delete_edges(edge_index_np, ind_test_idx)

        u_mask = data.val_mask.clone()

        idx_test = torch.from_numpy(ind_test_idx).long()
        u_trains_idx = torch.from_numpy(u_trains_idx).long()
        idx_unlabel = torch.cat([idx_val.view(-1), u_trains_idx.view(-1)], dim=0)

        test_mask[idx_test] = True
        u_mask[idx_unlabel] = True

    else:
        random_state = np.random.RandomState(0)
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels,
                                                                train_examples_per_class=train_examples_per_class,
                                                                val_examples_per_class=val_examples_per_class)
        train_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        val_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        test_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        u_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)

        idx_test_origin = idx_test.view(-1).numpy()
        ind_len = int(idx_test_origin.shape[0] * ind_ratio)
        if ind_len <= idx_test_origin.shape[0]:
            random_state = np.random.RandomState(0)
            ind_test_idx = random_state.choice(idx_test_origin, ind_len, replace=False)
        else:
            ind_test_idx = idx_test_origin

        u_trains_idx = np.array(list(set(idx_test_origin.tolist()).difference(set(ind_test_idx.tolist()))))
        # build indctive edge index
        edge_index_np = edge_index.numpy()
        edge_index_train = delete_edges(edge_index_np, ind_test_idx)

        idx_test = torch.from_numpy(ind_test_idx).long()
        u_trains_idx = torch.from_numpy(u_trains_idx).long()

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True
        u_mask[idx_val] = True
        u_mask[u_trains_idx] = True

    print('For {} dataset, we have #node: {} # feat: {} #classes: {} train_idx: {} val_inx: {} test_idx: {} un_idx: {}'
          .format(name, x.shape[0], x.shape[1], num_class, train_mask.sum().item(), val_mask.sum().item(),
                  test_mask.sum().item(), u_trains_idx.shape[0]))

    data = Data(x=x, y=y, edge_index=edge_index_train, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                u_mask=u_mask, edge_index_test=edge_index)
    return data


def get_dataset_split_inductive(path, name, train_examples_per_class, val_examples_per_class, ind_ratio):
    datasets = get_dataset(path, name)
    data = datasets[0]
    x, edge_index, y = data.x, data.edge_index, data.y.view(-1)
    num_class = torch.max(y) + 1
    ones_hot = torch.eye(num_class)
    labels = ones_hot[y].data.numpy()
    if name == 'Cora' or name == 'CiteSeer' or name == 'PubMed':
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = torch.zeros_like(data.test_mask, dtype=torch.bool)
        idx_train = torch.nonzero(train_mask.int())
        idx_val = torch.nonzero(val_mask.int())
        idx_test_origin = torch.nonzero(data.test_mask.int()).view(-1).numpy()

        non_test = set(torch.cat([idx_train, idx_val], dim=0).view(-1).data.numpy().tolist())
        all_test = set(list(range(x.shape[0])))
        idx_test = list(all_test.difference(non_test))
        ind_len = int(len(idx_test) * ind_ratio)
        if ind_len <= idx_test_origin.shape[0]:
            random_state = np.random.RandomState(0)
            ind_test_idx = random_state.choice(idx_test_origin, ind_len, replace=False)
        else:
            ind_test_idx = idx_test_origin

        u_trains_idx = np.array(list(set(idx_test).difference(set(ind_test_idx.tolist()))))
        # build indctive edge index
        edge_index_np = edge_index.numpy()
        edge_index_train = delete_edges(edge_index_np, ind_test_idx)

        u_mask = data.val_mask.clone()

        idx_test = torch.from_numpy(ind_test_idx).long()
        u_trains_idx = torch.from_numpy(u_trains_idx).long()
        idx_unlabel = torch.cat([idx_val.view(-1), u_trains_idx.view(-1)], dim=0)

        test_mask[idx_test] = True
        u_mask[idx_unlabel] = True

    else:
        random_state = np.random.RandomState(0)
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class=train_examples_per_class,
                                                                val_examples_per_class=val_examples_per_class)
        train_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        val_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        test_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        u_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)

        idx_test_origin = idx_test.view(-1).numpy()
        ind_len = int(idx_test_origin.shape[0] * ind_ratio)
        if ind_len <= idx_test_origin.shape[0]:
            random_state = np.random.RandomState(0)
            ind_test_idx = random_state.choice(idx_test_origin, ind_len, replace=False)
        else:
            ind_test_idx = idx_test_origin

        u_trains_idx = np.array(list(set(idx_test_origin.tolist()).difference(set(ind_test_idx.tolist()))))
        # build indctive edge index
        edge_index_np = edge_index.numpy()
        edge_index_train = delete_edges(edge_index_np, ind_test_idx)

        idx_test = torch.from_numpy(ind_test_idx).long()
        u_trains_idx = torch.from_numpy(u_trains_idx).long()

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True
        u_mask[idx_val] = True
        u_mask[u_trains_idx] = True

    print('For {} dataset, we have #node: {} # feat: {} #classes: {} train_idx: {} val_inx: {} test_idx: {} un_idx: {}'
          .format(name, x.shape[0], x.shape[1], num_class, train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item(), u_trains_idx.shape[0]))

    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, u_mask=u_mask, edge_index_train=edge_index_train)
    return data


def get_dataset_benchmark(path, name, train_examples_per_class, val_examples_per_class):
    datasets = get_dataset(path, name)
    data = datasets[0]
    x, edge_index, y = data.x, data.edge_index, data.y.view(-1)
    num_class = torch.max(y) + 1
    ones_hot = torch.eye(num_class)
    labels = ones_hot[y].data.numpy()
    if name == 'Cora' or name == 'CiteSeer' or name == 'PubMed':
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        idx_train = torch.nonzero(train_mask.int())
        idx_val = torch.nonzero(val_mask.int())
        non_test = set(torch.cat([idx_train, idx_val], dim=0).view(-1).data.numpy().tolist())
        all_test = set(list(range(x.shape[0])))
        idx_test = np.array(list(all_test.difference(non_test)))
        idx_test = torch.from_numpy(idx_test)
        test_mask[idx_test] = True
    else:
        random_state = np.random.RandomState(0)
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class=train_examples_per_class,
                                                                val_examples_per_class=val_examples_per_class)
        train_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        val_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)
        test_mask = torch.zeros_like(y.view(-1), dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

    print('For {} dataset, we have #node: {} # feat: {} #classes: {} train_idx: {} val_inx: {} test_idx: {}'
          .format(name, x.shape[0], x.shape[1], num_class, train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()))

    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data



def subgraph_sample(adj_t, split_idx, sample_size=500000):
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    train_len = train_idx.shape[0]
    valid_len = valid_idx.shape[0]

    torch.manual_seed(0)
    perm = torch.randperm(test_idx.size(0))
    perm = perm[0:sample_size]
    test_idx = test_idx[perm]

    new_index = torch.cat([train_idx, valid_idx, test_idx], dim=0)
    new_index_array = new_index.data.numpy()

    oldtonew_dict = {key: i for i, key in enumerate(new_index_array)}

    row = adj_t.storage.row().numpy()
    col = adj_t.storage.col().numpy()
    edge_index = []
    cnt = 1
    t1 = time.time()
    print('--- Start sampling for {} pints'.format(new_index_array.shape[0]))
    for index in new_index_array:
        neigh_index = np.where(row == index)[0]
        neigh_index = [col[i] for i in neigh_index if col[i] in new_index_array]
        neigh_preser = [[oldtonew_dict[index], oldtonew_dict[ele]] for ele in neigh_index]
        edge_index.extend(neigh_preser)
        if cnt % 50000 == 0:
            print('Run over {}/{} with time={}'.format(cnt, new_index_array.shape[0], time.time() - t1))
            t1 = time.time()
        cnt += 1
    # row = adj_t.storage.row().numpy()
    # valid_row = np.array([True if ele in new_index_array else False for ele in row])
    # col = adj_t.storage.col().numpy()
    # valid_col = np.array([True if ele in new_index_array else False for ele in col])
    # valid_all = valid_row & valid_col
    #
    #
    edge_index = np.array(edge_index).transpose()
    assert edge_index.shape[0] == 2

    edge_index = torch.from_numpy(edge_index)

    split_index = torch.arange(new_index.shape[0], dtype=train_idx.dtype)
    train_idx = split_index[0: train_len]
    valid_idx = split_index[train_len: train_len + valid_len]
    test_idx = split_index[train_len + valid_len:]

    return edge_index, new_index, train_idx, valid_idx, test_idx


def get_ogb_split(path, name, ind_ratio):
    if name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=name,
                                         transform=T.ToSparseTensor())

        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        # x, edge_index, y = data.x, data.adj_t, data.y.view(-1)
        x, y = data.x, data.y.view(-1)
        edge_index = torch.stack([data.adj_t.storage.row(), data.adj_t.storage.col()], dim=0)
        edge_index_test = torch.stack([data.adj_t.storage.row(), data.adj_t.storage.col()], dim=0)

    elif name == "ogbn-products":
        dataset = PygNodePropPredDataset(name='ogbn-products',
                                         transform=T.ToSparseTensor())
        data = dataset[0]

        split_idx = dataset.get_idx_split()
        sample_size = 500000
        x, edge_index, y = data.x, data.adj_t, data.y.view(-1)
        x_file = 'dataset/ogbn-products-x-{}.pt'.format(sample_size)
        y_file = 'dataset/ogbn-products-y-{}.pt'.format(sample_size)
        edge_file = 'dataset/ogbn-products-edge_index-{}.pt'.format(sample_size)
        trainidx_file = 'dataset/ogbn-products-train_index-{}.pt'.format(sample_size)
        valididx_file = 'dataset/ogbn-products-valid_index-{}.pt'.format(sample_size)
        testidx_file = 'dataset/ogbn-products-test_index-{}.pt'.format(sample_size)
        print('---Start sampling test dataset')
        if osp.exists(x_file) & osp.exists(y_file) & osp.exists(edge_file):
            print('Loading saved data...')
            x = torch.load(x_file)
            y = torch.load(y_file)
            edge_index = torch.load(edge_file)
            train_idx = torch.load(trainidx_file)
            valid_idx = torch.load(valididx_file)
            test_idx = torch.load(testidx_file)
        else:
            t1 = time.time()
            print('Preprocessing sampled data...')
            edge_index, new_index, train_idx, valid_idx, test_idx = subgraph_sample(data.adj_t, split_idx, sample_size=sample_size)
            x = x[new_index]
            y = y[new_index]
            torch.save(x, x_file)
            torch.save(y, y_file)
            torch.save(edge_index, edge_file)
            torch.save(train_idx, trainidx_file)
            torch.save(valid_idx, valididx_file)
            torch.save(test_idx, testidx_file)
            print('---End sampling test dataset with time={}'.format(time.time() - t1))

    else:
        dataset = PygNodePropPredDataset(
            name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
        data = dataset[0]

        # Move edge features to node features.
        data.x = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        x, edge_index, y = data.x, data.adj_t, data.y.view(-1)
    # data = datasets[0]
    # x, edge_index, y = data.x, data.edge_index, data.y.view(-1)

    # if name == 'ogbn-arxiv':
    idx_train = train_idx
    idx_val = valid_idx
    idx_test_origin = test_idx.view(-1).data.numpy()

    non_test = set(torch.cat([idx_train, idx_val], dim=0).view(-1).data.numpy().tolist())
    all_test = set(list(range(x.shape[0])))
    idx_test = list(all_test.difference(non_test))
    ind_len = int(len(idx_test) * ind_ratio)
    if ind_len <= idx_test_origin.shape[0]:
        random_state = np.random.RandomState(0)
        ind_test_idx = random_state.choice(idx_test_origin, ind_len, replace=False)
    else:
        ind_test_idx = idx_test_origin

    u_trains_idx = np.array(list(set(idx_test).difference(set(ind_test_idx.tolist()))))
    # build indctive edge index
    # edge_index_np = edge_index.numpy()
    # edge_index_np = torch.stack([edge_index.storage.row(), edge_index.storage.col()], dim=0).numpy()
    edge_index_np = edge_index.numpy()
    edge_index_train = delete_edges(edge_index_np, ind_test_idx)
    # edge_index = SparseTensor(row=edge_index_train[0], col=edge_index_train[1])
    assert edge_index_train.shape[0] == 2

    idx_test = torch.from_numpy(ind_test_idx).long()
    u_trains_idx = torch.from_numpy(u_trains_idx).long()
    idx_unlabel = torch.cat([idx_val.view(-1), u_trains_idx.view(-1)], dim=0)

    print('train_len={} valid_len={} test_len={}'.format(train_idx.shape[0], valid_idx.shape[0], test_idx.shape[0]))
    num_class = torch.max(y) + 1

    train_mask = torch.zeros_like(y, dtype=torch.bool)
    val_mask = torch.zeros_like(y, dtype=torch.bool)
    test_mask = torch.zeros_like(y, dtype=torch.bool)
    u_mask = torch.zeros_like(y, dtype=torch.bool)

    u_mask[valid_idx] = True
    u_mask[idx_unlabel] = True

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[idx_test] = True

    print('For {} dataset, we have #node: {} # feat: {} #classes: {} train_idx: {} val_inx: {} test_idx: {} un_idx: {}'
          .format(name, x.shape[0], x.shape[1], num_class, train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item(), u_mask.sum().item()))

    data = Data(x=x, y=y, edge_index=edge_index_train, edge_index_test=edge_index_test, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, u_mask=u_mask)
    return data


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code', 'ogbn-proteins']
    name = 'dblp' if name == 'DBLP' else name
    # root_path = osp.expanduser('~/datasets')
    root_path = path

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        if name == 'ogbn-proteins':
            dataset = PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.ToSparseTensor())
            data = dataset[0]
            x = data.adj_t.mean(dim=1)
            edge_index = torch.stack([data.adj_t.storage.row(), data.adj_t.storage.col()], dim=0)
            print('Data statistics for ogbn-proteins: n={} f={} #edges={}'.format(x.shape[0], x.shape[1], edge_index.size(1)))

            dataset = Data(x=x, edge_index=edge_index)
            return [dataset]
        else:
            return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))
    print(train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1
    train_indices = torch.LongTensor(train_indices)
    val_indices = torch.LongTensor(val_indices)
    test_indices = torch.LongTensor(test_indices)
    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    for key in sample_indices_per_class.keys():
        print('Key: {} has {} samples'.format(key, len(sample_indices_per_class[key])))
    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])
