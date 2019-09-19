# this is to optimize otcS12mdl model

from otcS12mdl import *
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval


baselS12space = {
    'module_config' : {
        'batch_size' : 512,
        'adam' : {
                    'lr' : hp.loguniform('lr', -50, 0)
        },
        'encoder': {
            'emb_acq_dims': {
                'state_id': (55, 3+hp.randint('state_id', 12)),
                'purpose_id': (5, 2),
                'mi_type_id': (4, 2),
                'occupancy_status_id': (4, 2), 
                'product_type_id': (2, 2), 
                'property_type_id': (6, 2), 
                'seller_id': (95, 3 + hp.randint('seller_id', 12)), 
                'zip3_id': (1001, 3 + hp.randint('zip3_id', 47))
            },
            'emb_seq_dims': {
                'yyyymm' : (219, 3 + hp.randint('yyyymm', 22)),
                'msa_id' : (407, 3 + hp.randint('msa_id', 22)),
                'servicer_id' : (46, 3 + hp.randint('servicer_id', 5))
            },
            'lstm' : {
                'lstm_size': (100 + hp.randint('lstm_size', 600)),
                'lstm_layers': (2 + hp.randint('lstm_layers', 3)),
                'lstm_dropout': hp.uniform('lstm_dropout', 0, 1)
            }
        },
        'pipe' : {
            'lin1' : 50 + hp.randint('lin1', 100),
            'lin_drp1' : hp.uniform('lin_drp1', 0, 1),
            'lin2' : 50 + hp.randint('lin2', 200),
            'lin_drp2' : hp.uniform('lin_drp2', 0, 1),
        }
    }
}

def loadDataset(PATH, oneChunkOnly=True, ratio=0):
    acq, idx_to_seq, seq, _, ym2idx = load_data(
        PATH, 
        verbose=True, 
        oneChunkOnly=oneChunkOnly)
    return FNMDatasetS12 (acq, idx_to_seq, seq, ym2idx, ratio)


if __name__ == "__main__":
    DEBUG = True
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    MODEL_PATH = '/home/user/notebooks/data/model/otcS12/'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    train_ds = loadDataset(TRAIN_PATH, oneChunkOnly=True, ratio=3)
    valid_ds = loadDataset(VALID_PATH, oneChunkOnly=True, ratio=3)

    def objective(args):
        module_config = args['module_config']
        model = OTCS12Model(module_config)
        model.dataLoaderTrain(train_ds, NUM_WORKERS=6)
        model.dataLoaderValid(valid_ds, NUM_WORKERS=6)
        model.useGPU(not DEBUG)
        model.makeParallel(not DEBUG)
        return model.fit(NUM_EPOCHS=1, save_model=False)


    best = fmin(objective, baselS12space, algo=tpe.suggest, max_evals=100)
    print(best)

# save best