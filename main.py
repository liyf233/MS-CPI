import torch
import numpy as np
import random
import os
import time

from tqdm import tqdm
from model import *
import timeit
from data import Mol2HeteroGraph
from rdkit import Chem
from torch.utils.tensorboard import SummaryWriter


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "GPCR"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('dataset/' + DATASET + '/word2vec_30/')
    smiles = np.load(dir_input + 'smiles.npy', allow_pickle=True)
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    index = [True] * smiles.shape[0]
    graphs = []
    i = 0
    for j, smi in enumerate(tqdm(smiles)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                exit(0)
            else:
                g = Mol2HeteroGraph(mol)
                if g.num_nodes('a') == 0:
                    index[j] = False
                    i += 1
                else:
                    graphs.append(g)
        except Exception as e:
            exit(0)
    print("ingore " + str(i))
    graphs = [graph.to(device) for graph in graphs]
    proteins2, interactions2 = [], []
    for idx, item in enumerate(index):
        if item == True:
            proteins2.append(proteins[idx])
            interactions2.append(interactions[idx])

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(graphs, proteins2, interactions2))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_2 = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_2, 0.5)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 42
    hid_dim = 400
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 2
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 300
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output/result/'+ DATASET + '.txt'
    file_model = 'output/model/' + 'celegans'
    file_decoder = 'output/model/' + 'compoumds_human.pth'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\t AUC_dev\t PRC_dev\tprecision_dev\trecall_dev\tAUC_test\tF1_dev\tPRC_test\t'+
                'precision_test\tACC_dev\trecall_test\tF1_test\tACC_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
        
    writer = SummaryWriter('logs/log1')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    max_AUC_test = 0
    total_step = 0
    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, step = trainer.train(dataset_train, device, writer, total_step)
        total_step = step
        AUC_dev, PRC_dev,precision_dev,recall_dev,F1_dev,ACC_dev = tester.test(dataset_dev)
        AUC_test, PRC_test,precision_test,recall_test,F1_test,ACC_test = tester.test(dataset_test)
        writer.add_scalars('train', {'total loss': loss_train}, epoch)
        writer.add_scalars('valid', {'AUC_dev': AUC_dev}, epoch)
        writer.add_scalars('valid', {'PRC_dev': PRC_dev}, epoch)
        writer.add_scalars('valid', {'precision_dev': precision_dev}, epoch)
        writer.add_scalars('valid', {'recall_dev': recall_dev}, epoch)
        writer.add_scalars('valid', {'F1_dev': F1_dev}, epoch)
        writer.add_scalars('valid', {'ACC_dev': ACC_dev}, epoch)
        writer.add_scalars('test', {'AUC_test': AUC_test}, epoch)
        writer.add_scalars('test', {'PRC_test': PRC_test}, epoch)
        writer.add_scalars('test', {'precision_test': precision_test}, epoch)
        writer.add_scalars('test', {'recall_test': recall_test}, epoch)
        writer.add_scalars('test', {'F1_test': F1_test}, epoch)
        writer.add_scalars('test', {'ACC_test': ACC_test}, epoch)
        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, PRC_dev,precision_dev,recall_dev,AUC_test,F1_dev, PRC_test,
                precision_test,ACC_dev,recall_test,F1_test,ACC_test]
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_test > max_AUC_test:
            # 先获取所有子模块的名称和对象
            modules = model.named_children() 
            # 找到 layer2 模块
            layer2 = None 
            for name, module in modules:
                if name == "c_encoder":
                    layer2 = module  
            # 保存单独的 layer2
            torch.save(layer2.state_dict(), 'output/model/' + 'Pharm.pth')
            torch.save(model.state_dict(), file_decoder)
            tester.save_model(model, file_model)

            max_AUC_test = AUC_test
        print('\t'.join(map(str, AUCs)))
    writer.close()

