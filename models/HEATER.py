# =============================================================================
# File: HEATER.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 15.3.2025
# Description: Script to run HEATER model.
# Notes: Modified as part of the bachelor's thesis work. Evaluation changed to user-split.
#        Different data loading was used and additionaly prediction mechanism was added.
# =============================================================================


import os
import argparse

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true', '1'):
        return True
    elif v.lower() in ('False', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--device", default=None, type=str, help="Limit device to run on")
parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")
parser.add_argument("--validation", default=False, type=str2bool, help="Use validation split: true/false")
# lr
parser.add_argument("--scheduler", default="none", type=str, help="Scheduler: CosineDecay or none")
parser.add_argument("--lr", default=.001, type=float, help="Learning rate for model training, only if scheduler is none")
parser.add_argument("--epochs", default=10, type=int, help="Training epochs, only if scheduler is none ")
parser.add_argument("--evaluate_epoch", default=False, type=str2bool, help="Evaluate after each epoch, only if scheduler is none")

parser.add_argument("--init_lr", default=.0, type=float, help="starting lr, only if scheduler is CosineDecay")
parser.add_argument("--warmup_lr", default=.002, type=float, help="max warmup lr, only if scheduler is CosineDecay")
parser.add_argument("--warmup_epochs", default=1, type=int, help="Warmup epochs, only if scheduler is CosineDecay")
parser.add_argument("--decay_epochs", default=5, type=int, help="Decay epochs, only if scheduler is CosineDecay")

# dataset
parser.add_argument("--dataset", default="-", type=str, help="Dataset to run on")
parser.add_argument("--use_cold_start", default=False, type=str2bool, help="Use cold start evaluation, default false")
parser.add_argument("--use_time_split", default=False, type=str2bool, help="Use time split evaluation, default false")

# sentence transformer details
parser.add_argument("--sbert", default=None, type=str, help="Input sentence transformer model to train")
parser.add_argument("--max_seq_length", default=0, type=int, help="Maximum sequece length for sbert")
# use images
parser.add_argument("--images", default=False, type=str2bool, help="Do we want to use images? [true/false]")
# use features
parser.add_argument("--features", default=False, type=str2bool, help="Do we want to use categorical features? [true/false]")

# ------------------------------------------------------------------------------------------------------------
parser.add_argument("--audio_att_model", default=False, type=str2bool, help="Whether to use audio attributes model, default false") 
parser.add_argument("--pooling_time", default="end", type=str, help="Wheater to use attributes, that were created through mean of results, or through mean of embeddings and then classified. [end/emb]")
parser.add_argument("--which_attributes", default="all", type=str, choices=["all", "library", "ast"], help="Which attributes to use? Options: all, library, ast")
parser.add_argument("--scale_AST_att", default=False, type=str2bool, help="Scale AST attributes? [true/false]")
parser.add_argument("--standardize_ess_att", default=True, type=str2bool, help="Standardize essential attributes? [true/false]")
parser.add_argument("--standardize_ast_att", default=True, type=str2bool, help="Standardize ast attributes? [true/false]")
# ------------------------------------------------------------------------------------------------------------
parser.add_argument("--audio_emb_model", default=False, type=str2bool, help="Whether to use audio embeddings model, default false")
parser.add_argument("--standardize", default=True, type=str2bool, help="Standardize the input? [true/false]")
# ------------------------------------------------------------------------------------------------------------

# model hyperparams
parser.add_argument("--factors", default=None, type=int, help="number of final factors. Default is half of ALS factors")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size of sampled users per training step")

# als pre training
parser.add_argument("--als_factors", default=1024, type=int, help="ALS mf factors")
parser.add_argument("--als_regularization", default=50, type=float, help="ALS mf regularization")
parser.add_argument("--als_iterations", default=20, type=int, help="ALS mf iterations")
parser.add_argument("--als_num_threads", default=0, type=int, help="ALS mf num threads to use")
parser.add_argument("--als_use_gpu", default=False, type=str2bool, help="ALS mf whether to use gpu")

args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

# limit visible devces for pytorch
if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

# force the usage of pytorch backend in keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import math 
import numpy as np
import torch

from baselines import ALSMatrixFactorizer
from dataloaders import beeformerDataset

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


from utils import *
from _datasets.utils import *
from config import config
from _datasets.pydatasets import SparseRecSysDatasetWithNegatives
from Audio_processor.Attributes_processor import Attributes_Processor, Embedding_Processor
from callbacks import Callbacks_HEATER

import subprocess
from time import time

torch.set_float32_matmul_precision('medium')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device('cpu')
print(f"Using device {DEVICE}")


class trainMFDataset(keras.utils.PyDataset):
    """
    input sparse interaction matrix + item_ids to know order of items
    output batches of user vectors and user ids
    """
    def __init__(self, X, device, batch_size=128, shuffle=True, workers=1, use_multiprocessing=False, max_queue_size=10):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing,
                         max_queue_size=max_queue_size)
        self.X = X
        self.user_indices = np.arange(self.X.shape[0])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.user_indices)

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.X.shape[0] / (self.batch_size))

    def __getitem__(self, n):
        ind = n*self.batch_size
        ind_min = ind
        ind_max = ind+self.batch_size
        slicer = self.user_indices[ind_min:ind_max]
        M = self.X[slicer]
        #R = torch.from_numpy(M.toarray().astype("float32")).cuda()
        R = M.toarray().astype("float32")
        #user_indices = self.user_ids[ind_min:ind_max]
        #slicer = torch.from_numpy(slicer).long().to(self.device)
        return slicer, torch.from_numpy(R).to(self.device)

class Heater(keras.models.Model):
    """
    A model for a recommendation system that uses matrix factorization techniques. 
    This model receives data from `SparseRecSysDataset`, where data consists of user vectors and a slicer for non-zero entries.

    Attributes:
        device (str): The device (e.g., "cpu" or "cuda") to run the model on.
        dataset (object): The dataset used to create the recommendation system.
        item_idx (array-like): The list of item indices, used for mapping back to the dataframe.
        U (torch.Tensor): The user embedding matrix.
        V (torch.Tensor): The item embedding matrix.
        B (torch.Tensor): Bias terms for items.
        in_layer_items (keras.layers.Dense): The input layer for item transformation.
        out_layer_items (keras.layers.Dense): The output layer for item transformation.
        out_layer_users (keras.layers.Dense): The output layer for user transformation.
    """
    def __init__(self, U, V, B, device, dataset, out_factors=None, item_idx=None):
        super().__init__()
        #self.A = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])))
        self.device = device
        self.dataset = dataset
        self.item_idx = item_idx
        #self.user_idx = user_idx 
        #self.input_items_idx = input_items_idx
        #self.output_items_idx = output_items_idx
        self.U = torch.from_numpy(U).to(device)
        self.V = torch.from_numpy(V).to(device)
        self.B = torch.from_numpy(B).to(device)
        self.U.requires_grad=False
        self.V.requires_grad=False
        self.B.requires_grad=False

        hidden = U.shape[1]
        if out_factors is None:
            out_factors = hidden // 2
            
        # items transformation 
        self.in_layer_items = keras.layers.Dense(hidden, activation="linear")
        self.in_layer_items.to(self.device)
        
        # final items transform
        self.out_layer_items = keras.layers.Dense(out_factors, activation="linear")    
        self.out_layer_items.to(self.device)
        
        # finel users transforms
        self.out_layer_users = keras.layers.Dense(out_factors, activation="linear")  
        self.out_layer_users.to(self.device)
        
        self.to(self.device)
        self.track_module_parameters()
        self(torch.from_numpy(np.arange(10, dtype="int64")))

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable( #keras.backend.Variable(
                initializer=param, trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)
        self.built = True
    
    def call(self, slicer, training=False):
        slicer = slicer.long()        
        U = self.U[slicer]

        Ii = self.in_layer_items(self.B)
        
        if training:
            additional_loss = NMSE(Ii, self.V)
            additional_loss = torch.mean(additional_loss)
            self.add_loss(additional_loss*0.01)
        
        Pi = self.out_layer_items(Ii)
        
        Pu = self.out_layer_users(U)
        output = Pu@(Pi.T)

        return output
        
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        #x = x.to(self.device)
        
        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()
        y_pred = self(x, training=True)  # Forward pass
        
        # Compute loss
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def predict_batch(self, U):
        Ii = self.in_layer_items(self.B)
        Pi = self.out_layer_items(Ii)  
        Pu = self.out_layer_users(U)
        output = Pu@(Pi.T)

        return output
    
    def predict_df(self, df, k=100, batch_size=1000, als_model=None):
        X_test = get_sparse_matrix_from_dataframe(df, item_indices=self.item_idx)
        
        # compute testing user embeddings
        test_user_embeddings = als_model.model.recalculate_user(
            userid=np.arange(X_test.shape[0]), 
            user_items=X_test
        )
        
        if args.als_use_gpu:
            test_user_embeddings = test_user_embeddings.to_numpy()
        
        U = torch.from_numpy(test_user_embeddings).to(DEVICE)
        
        n_batches = ceil(X_test.shape[0]/batch_size)
        uids = df.user_id.cat.categories.to_numpy()
        
        dfs=[]
        for i in tqdm(range(n_batches)):
            i_min = i*batch_size
            i_max = i_min+batch_size
            batch=X_test[i_min:i_max].toarray()
            
            i_max = min(i_min + batch_size, X_test.shape[0])
                 
            #slicer = torch.arange(i_min, i_max, device=self.device)
            with torch.no_grad():
                preds = self.predict_batch(U[i_min:i_max])
            
            batch_tensor = torch.from_numpy(batch).to(preds.device)  # Convert batch to a PyTorch tensor and move it to the same device as preds
            preds = preds * (1 - batch_tensor)
            
            batch_uids = uids[i_min:i_max]
            values_, indices_ = torch.topk(preds.to("cpu"), k)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*k).flatten("F"), "item_id": np.array(
                self.item_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)            
        return pd.concat(dfs)

def NMSE(x,y):
    x=torch.nn.functional.normalize(x, dim=-1)
    y=torch.nn.functional.normalize(y, dim=-1)
    return keras.losses.mean_squared_error(x,y)


def main(args):
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    
    vargs["cuda_or_cpu"]=DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.dataset not in config.keys():
        print("Unknown dataset. List of available datsets: \n")
        for x in config.keys():
            print(x)
        return 

    dataset, params = config[args.dataset]
    dataset.load_interactions(**params)
    
    print(dataset)


    if args.use_time_split:
        evaluator_class = TimeBasedEvaluation
    elif args.use_cold_start:
        evaluator_class = ColdStartEvaluation
    else:  # Default: user-based split strategy
        evaluator_class = Evaluation
        
    test_evaluator = evaluator_class(dataset, "test")
        
    if args.validation:
        val_evaluator = evaluator_class(dataset, "validation")
    else:
        val_evaluator = None  
    
    
    print(dataset)

    if args.validation:
        X = get_sparse_matrix_from_dataframe(dataset.train_interactions)        
        train_items_idx=dataset.train_interactions.item_id.cat.categories
    else:
        X = get_sparse_matrix_from_dataframe(dataset.full_train_interactions)
        train_items_idx=dataset.full_train_interactions.item_id.cat.categories

    if args.sbert != None:
        sbert = SentenceTransformer(args.sbert, device=DEVICE)
        if args.max_seq_length>0:
            sbert.max_seq_length = args.max_seq_length
            embs_full = sbert.encode(dataset.texts, show_progress_bar=True)

    if args.images:
        embs_full = dataset.image_embeddings 
    
    if args.features:
        embs_full = dataset.features_embeddings 
    
    # ****************************************************************************************************************************    
    if args.audio_emb_model:
        prepr_config = {
            "embeddings_json_path": dataset.embeddings_json_path,
            "standardize": args.standardize,
        }
        processor = Embedding_Processor(prepr_config, dataset.train_interactions.item_id.cat.categories)
        embs_full = processor.get_ordered_tensor(dataset.all_interactions.item_id.cat.categories)
    
    if args.audio_att_model:
        prepr_config = {
            "csv_path": dataset.csv_path_end_pooling if args.pooling_time == "end" else dataset.csv_path_emb_pooling,
            "pooling_time": args.pooling_time,
            "AST_occurence_path": dataset.AST_occurence_path,
            "standardize_ess_att": args.standardize_ess_att,
            "standardize_ast_att": args.standardize_ast_att,
            "scale_AST_att": args.scale_AST_att,
            "which_attributes": args.which_attributes,
        }
        processor = Attributes_Processor(prepr_config, dataset.train_interactions.item_id.cat.categories)
        embs_full = processor.get_ordered_tensor(dataset.all_interactions.item_id.cat.categories)
    # ****************************************************************************************************************************
        
        
    print("creating als model")
    
    als_model = ALSMatrixFactorizer( 
        factors=args.als_factors,
        regularization=args.als_regularization,
        iterations=args.als_iterations,
        #use_gpu=False,
        use_gpu=args.als_use_gpu,
        num_threads=args.als_num_threads,
        item_idx=train_items_idx, 
    )
        
    print("training als model")

    als_model.fit(X)
    if args.als_use_gpu:
        als_item_embeddings = als_model.model.item_factors.to_numpy()
        als_user_embeddings = als_model.model.user_factors.to_numpy()
    else:
        als_item_embeddings = als_model.model.item_factors
        als_user_embeddings = als_model.model.user_factors
    
    # ********************************************************************
    iids_full = dataset.all_interactions.item_id.cat.categories
    # I dont initialize items_texts
    #iids_full = dataset.items_texts.item_id.to_list()
    # ********************************************************************
    piids = pd.Index(iids_full)

    source_slicer=np.array([piids.get_loc(x) for x in train_items_idx])
    source_embs = embs_full[source_slicer]

    #target_slicer = [piids.get_loc(x) for x in test_evaluator.test_target.item_id.cat.categories]
    #target_embs = embs_full[target_slicer]
    data_loader = trainMFDataset(X, device=DEVICE, batch_size=args.batch_size)

    model = Heater(
        U=als_user_embeddings,
        V=als_item_embeddings,
        B=source_embs,
        device=DEVICE,
        dataset=dataset,
        out_factors=256,
        item_idx=train_items_idx,
    )
    model.to(DEVICE)
     
    print(model)
    
    if args.scheduler == "CosineDecay":
        scheduler = keras.optimizers.schedules.CosineDecay(
                args.init_lr,
                args.decay_epochs*len(data_loader),
                alpha=0.0,
                name="CosineDecay",
                warmup_target=args.warmup_lr,
                warmup_steps=args.warmup_epochs*len(data_loader),
            )
        epochs = args.decay_epochs+args.warmup_epochs
    else:
        scheduler = args.lr
        epochs = args.epochs

    #model.compile(optimizer=NadamS(learning_rate=scheduler), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()])
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=scheduler), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()])
    model.train_step(data_loader[0])
    
    fits = []
    # evaluation
    #model.B = torch.from_numpy(target_embs).to(DEVICE)
    
    cbs = []
    if args.evaluate_epoch:
        eval_cb = Callbacks_HEATER(
            als_model=als_model,
            evaluator=val_evaluator if args.validation else test_evaluator,
            logdir=folder,
            DEVICE=DEVICE,
            validation=args.validation,
            evaluate_epoch=args.evaluate_epoch,
        )
        cbs.append(eval_cb)
    
    # training
    print(f"Training for {args.epochs} epochs.")
    start = time()
    f = model.fit(data_loader, epochs=epochs, callbacks=cbs, verbose=1)
    train_time = time()-start
    
    print("training finished")
    
    # if validation is used and evaluate_epoch is not set, we need to evaluate the model on the validation set
    if args.validation and not args.evaluate_epoch:
        df_preds = model.predict_df(val_evaluator.test_src, als_model = als_model)
        results = val_evaluator(df_preds)

        df = pd.DataFrame([results])
        df["epoch"] = epochs
        df = df[list(df.columns[-1:]) + list(df.columns[:-1])]  # Reorder columns

        save_to = folder + "/val_logs.csv"
        file_exists = os.path.isfile(save_to)
        df.to_csv(save_to, mode="a", header=not file_exists, index=False)
    
    
    # evaluation on testing data
    df_preds = model.predict_df(test_evaluator.test_src, als_model = als_model)
    results=test_evaluator(df_preds)

    print(results)
  
    ks = list(f.history.keys())    
    dc = {k:np.array([(f.history[k]) for f in fits]).flatten() for k in ks}
    dc["epoch"] = np.arange(len(dc[list(dc.keys())[0]]))+1
    df = pd.DataFrame(dc)
    df[list(df.columns[-1:])+list(df.columns[:-1])]
    
    df.to_csv(f"{folder}/history.csv")
    print("history file written")

    pd.Series(results).to_csv(f"{folder}/result.csv")
    print("results file written")
    
    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")


if __name__ == "__main__":
    main(args)