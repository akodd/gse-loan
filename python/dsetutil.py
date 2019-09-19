import numpy as np
import pickle
import tqdm
import glob

def load_data(data_path, verbose=True, oneChunkOnly=False):
    acquisition_nname = data_path + '/fnm_input_acq.npy'
    sequence_nname = data_path + '/fnm_input_seq_*.npy' 
    idx_to_seq_nname = data_path + '/fnm_input_idx_to_seq.npy'
    macros_nname = data_path + '/fred_norm.npy'
    ym2idx_nname = data_path + '/ym2idx.pickle'
    
    if verbose:
        print('Acquisition: {}'.format(acquisition_nname))
        print('Sequence: {}'.format(sequence_nname))
        print('Index to Sequence Index: {}'.format(idx_to_seq_nname))
        print('Macros: {}'.format(macros_nname))
        print('YYYYMM to Index: {}'.format(ym2idx_nname))
 
    acq = np.load(acquisition_nname)
    idx_to_seq = np.load(idx_to_seq_nname)
    macros = np.load(macros_nname)
    #ym2idx = np.load(ym2idx_nname)
    with open(ym2idx_nname, 'rb') as f:
        ym2idx = pickle.load(f)

    seq_files = sorted([f for f in glob.glob(sequence_nname, recursive=False)])
    seq = [None] * len(seq_files)
    
    if verbose:
        print("loading seq_chunk")
        itt = tqdm.tqdm(seq_files)
    else:
        itt = seq_files
    for chunk_idx, seq_chunk in enumerate(itt):
        if verbose:
            itt.set_description('Loading: ' + seq_chunk)
        seq[chunk_idx] = np.load(seq_chunk)
        if oneChunkOnly:
            break
    
    if oneChunkOnly:
        x = idx_to_seq[:, 0]==0
        idx_to_seq = idx_to_seq[x, :]
        acq = acq[x, :]
        
    return acq, idx_to_seq, seq, macros, ym2idx

if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    TEST_PATH  = '/home/user/notebooks/data/test'

    def load_report(data_path):
        acq, idx_to_seq, seq, macro, ym2idx = load_data(data_path, True)
        print('acq.shape={}'.format(acq.shape))
        print('idx_to_seq.shape={}'.format(idx_to_seq.shape))
        print('macro.shape={}'.format(macro.shape))
        print('len(ym2idx)={}'.format(len(ym2idx)))

        for i in range(len(seq)):
            print('\tseq[{}].shape={}'.format(i, seq[i].shape))


    print('TRAIN DATA #######')
    load_report(TRAIN_PATH)
    print('VALID DATA #######')
    load_report(VALID_PATH)
    print('TEST DATA #######')
    load_report(TEST_PATH)