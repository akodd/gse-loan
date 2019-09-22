# this is to optimize otcS12mdl model

from otcS12mdl import *
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval


baselS12space = {
    'module_config' : {
        'seq_n_features' : 25,
        'batch_size' : 256,
        'adam' : {
            'lr' : hp.loguniform('lr', -10, 0)
        },
        'encoder': {
            'emb_acq_dims': {
                'state_id': (55, 3+hp.randint('state_id', 12)),
                'purpose_id': (5, 2),
                'mi_type_id': (4, 2),
                'occupancy_status_id': (4, 2), 
                'product_type_id': (2, 2), 
                'property_type_id': (6, 2), 
                'seller_id': (95, 3 + hp.randint('seller_id', 47)), 
                'zip3_id': (1001, 3 + hp.randint('zip3_id', 47))
            },
            'emb_seq_dims': {
                'yyyymm' : (219, 3 + hp.randint('yyyymm', 47)),
                'msa_id' : (407, 3 + hp.randint('msa_id', 47)),
                'servicer_id' : (46, 3 + hp.randint('servicer_id', 5))
            },
            'lstm' : {
                'lstm_size': (100 + hp.randint('lstm_size', 1000)),
                'lstm_layers': (2 + hp.randint('lstm_layers', 6)),
                'lstm_dropout': hp.uniform('lstm_dropout', 0, 1)
            }
        },
        'pipe' : {
                'l1' : (50 + hp.randint('l1_l', 500), 
                        hp.uniform('l1_d', 0, 1)),
                'l2' : (50 + hp.randint('l2_l', 500), 
                        hp.uniform('l2_d', 0, 1))
        }
    }
}

if __name__ == "__main__":
    DEBUG = True
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    MODEL_PATH = '/home/user/notebooks/data/model/otcS12/'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    train_ds = loadDataset(TRAIN_PATH, dlq_dim=19, oneChunkOnly=True, ratio=5)
    valid_ds = loadDataset(VALID_PATH, dlq_dim=19, oneChunkOnly=True, ratio=5)

    def objective(args):
        module_config = args #['module_config']
        print(module_config)
        model = OTCS12Model(module_config)
        model.dataLoaderTrain(train_ds, NUM_WORKERS=6)
        model.dataLoaderValid(valid_ds, NUM_WORKERS=6)
        model.useGPU(True, verbose=False)
        model.makeParallel(True, verbose=False)
        valid_error = model.fit(NUM_EPOCHS=3, save_model=False)
        return valid_error


    best = fmin(objective, baselS12space, algo=tpe.suggest, max_evals=1000)
    print(best)

# save best