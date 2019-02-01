import numpy as np 

def batch_iter(train_datasets, batch_size, num_epoch, shuffle=False):
    train_datasets = np.array(train_datasets)
    total_batch_num = int(len(train_datasets) / batch_size) + 1
    for epoch in range(num_epoch):
        if shuffle:
            shuffled_index = np.random.permutation(np.arange(len(train_datasets)))
            train_datasets = train_datasets[shuffled_index]
        for i in range(total_batch_num):
            start_idx = i * batch_size
            last_idx = min((i + 1) * batch_size, len(train_datasets) - 1)
            yield train_datasets[start_idx:last_idx]