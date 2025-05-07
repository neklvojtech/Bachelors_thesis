# =============================================================================
# File: train.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 26.3.2025
# Description: Script to train the Beeformer model.
# Notes: Modified as part of the bachelor's thesis work. 
#        Compatibility with audio and featurs beeFormer was added.
# =============================================================================

import os
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true', '1'):
        return True
    elif v.lower() in ('False', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# need to parse arguments before loading pytorch
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed, default 42")
parser.add_argument("--device", default=None, type=str, help="Limit device to run on, default None (no limit)")
parser.add_argument(
    "--devices", default=None, type=str, help="Devices for multi-device training ex. [0,1,2,3], default None "
)
parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")
parser.add_argument("--validation", default=False, type=str2bool, help="Use validation split: true/false, default false")

# lr
parser.add_argument(
    "--lr", default=1e-5, type=float, help="Learning rate for model training, only if scheduler is none, default 1e-5"
)
parser.add_argument(
    "--scheduler", default="none", type=str, help="Scheduler: LinearWarmup, CosineDecay or none, default none"
)
parser.add_argument("--init_lr", default=0.0, type=float, help="starting lr, only if scheduler is not none, default 0")
parser.add_argument(
    "--warmup_lr", default=1e-4, type=float, help="max warmup lr, only if scheduler is not none, default 1e-4"
)
parser.add_argument(
    "--target_lr", default=1e-6, type=float, help="final lr, only if scheduler is LinearWarmup, default 1e-6"
)
parser.add_argument(
    "--warmup_epochs", default=1, type=int, help="Warmup epochs, only if scheduler is not none, default 1"
)
parser.add_argument(
    "--decay_epochs", default=3, type=int, help="Decay epochs, only if scheduler is not none, default 3"
)
parser.add_argument(
    "--tuning_epochs", default=1, type=int, help="Final lr epochs, only if scheduler is LinearWarmup, default 1"
)
parser.add_argument("--epochs", default=5, type=int, help="Training epochs, only if scheduler is none, default 5")

# dataset
parser.add_argument("--dataset", default="-", type=str, help="Dataset to run on")
parser.add_argument("--use_cold_start", default=False, type=str2bool, help="Use cold start evaluation, default false")
parser.add_argument("--use_time_split", default=False, type=str2bool, help="Use time split evaluation, default false")
parser.add_argument(
    "--prefix",
    default=None,
    type=str,
    help="Add prefix to every item description (example for e5 models add query: as prefix to every item description - see https://huggingface.co/intfloat/multilingual-e5-base#faq), default None",
)

# sentence transformer details
parser.add_argument("--sbert", default=None, type=str, help="Input sentence transformer model to train")
parser.add_argument("--image_model", default=None, type=str, help="Input image model model to train")

# ------------------------------------------------------------------------------------------------------------
parser.add_argument("--audio_model", default=None, type=str, help="Input audio model to train") 
parser.add_argument("--token_pooling", default="CLS", type=str, help="Pooling for tokens of audio model, default cls")
parser.add_argument("--num_spectograms", default=6, type=int, help="Number of spectograms to take from audio, default 6") # 1 minute of song
parser.add_argument("--more_spec_pooling", default="mean", type=str, help="Pooling for multiple spectograms ( mean x max), default mean")
parser.add_argument("--num_layers_to_freeze", default=None, type=int, help="Number of AST layers to freeze, default None (no freeze)")
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
# NN configuration
# if audio_att_model or audio_emb_model is used
parser.add_argument("--num_hidden_layers", default=3, type=int, help="Number of layers in Neural Network, default 2",)
parser.add_argument("--hidden_dimension", default=1024, type=int, help="Hidden dimension of Neural Network, default 1024")
parser.add_argument("--output_dimension", default=768, type=int, help="Output dimension of Neural Network, default 768")
# ------------------------------------------------------------------------------------------------------------

parser.add_argument(
    "--max_seq_length",
    default=None,
    type=int,
    help="Maximum sequence length, default None (use original value from sbert)",
)
parser.add_argument(
    "--preproces_html",
    default=False,
    type=str2bool,
    help="whether to get rid of html inside descriptions (not relevant for LLM generated descriptions), default false",
)

# model hyperparams
parser.add_argument(
    "--max_output",
    default=10000,
    type=int,
    help="Max number of items on output (m parameter from paper), default 10000",
)
parser.add_argument(
    "--batch_size", default=1024, type=int, help="Batch size of sampled users per training step, default 1024"
)
parser.add_argument(
    "--top_k",
    default=0,
    type=int,
    help="Optimize only for top-k predictions on the output of the model. May bring some improvement for large, sparse datasets (in theory). Default 0 (not use)",
)
parser.add_argument(
    "--sbert_batch_size",
    default=200,
    type=int,
    help="Batch size for computing embeddings with sentence transformer, default 200",
)
# output model name
parser.add_argument(
    "--model_name",
    default="my_model",
    type=str,
    help="Output sentence transformer model name to train, default my_model",
)

# evaluate
parser.add_argument(
    "--evaluate", default=False, type=str2bool, help="final evaluation after training [true/false], default false"
)
parser.add_argument(
    "--evaluate_epoch", default=False, type=str2bool, help="evaluation after every epoch [true/false], default false"
)
parser.add_argument(
    "--save_every_epoch", default=True, type=str2bool, help="save after every epoch [true/false], default true"
)

args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

# limit visible devces for pytorch
if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

# force the usage of pytorch backend in keras
os.environ["KERAS_BACKEND"] = "torch"

# now we can finally import modules

import keras
import math
import numpy as np
import sentence_transformers
import subprocess
import time
import torch

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from callbacks import evaluateWriter
from config import config
from dataloaders import beeformerDataset
from models import NMSEbeeformer, SparseKerasELSA  # , simpleBee
from schedules import LinearWarmup
from utils import *

# ------------------------------------------------------------------------------------------------------------
import audio 
import audio_att
# ------------------------------------------------------------------------------------------------------------

torch.set_float32_matmul_precision('medium')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

def load_data(args):
    if args.validation:
        what = "val"
    else:
        what = "test"

    # read data
    if args.dataset in config.keys():
        dataset, params = config[args.dataset]
        dataset.load_interactions(**params)     

        
        # ******* big change, when validation flag is used, its used for_val_logs, but the final evaluation is on testing set
        if args.use_time_split:
            evaluator_class = TimeBasedEvaluation
        elif args.use_cold_start:
            evaluator_class = ColdStartEvaluation
        else:  # Default: user-based split strategy
            evaluator_class = Evaluation
        
        evaluator = evaluator_class(dataset, "test")
        
        if args.validation:
            val_evaluator = evaluator_class(dataset, "validation")
        else:
            val_evaluator = None

        # -------------------------------------------------------------------------------------------------------------------------
        if "items_raw_data" in params:
            items_d = dataset.items_texts
            items_d["asin"] = items_d.item_id
        else:
            items_d = None
        # -------------------------------------------------------------------------------------------------------------------------
        # items_d = dataset.items_texts
        # items_d["asin"] = items_d.item_id
        # -------------------------------------------------------------------------------------------------------------------------
        if args.validation:
            _train_interactions = dataset.train_interactions
        else:
            _train_interactions = dataset.full_train_interactions

    elif args.dataset == "goodlens":
        # todo: should be rewritten to combine any two datasets (not as simple as it looks)
        dataset, params = config["ml20m"]
        dataset.load_interactions(**params)

        if args.use_cold_start:
            evaluator = ColdStartEvaluation(dataset, what=what)
        elif args.use_time_split:
            evaluator = TimeBasedEvaluation(dataset, what=what)
        else:
            evaluator = Evaluation(dataset, what=what)

        dataset2, params2 = config["goodbooks"]
        dataset2.load_interactions(**params2)
        if args.use_cold_start:
            # this must be done, because init in evaluator is modifiyng dataset object
            # it also mean that evaluation will be eventually done on movielens
            evaluator2 = ColdStartEvaluation(dataset2)
        else:
            evaluator2 = Evaluation(dataset2)

        # merge the two datasets
        if args.validation:
            df = dataset.train_interactions.copy()
        else:
            df = dataset.full_train_interactions.copy()

        it = dataset.items_texts.copy()
        df["user_id"] = df.user_id.apply(lambda x: "m" + x)
        df["item_id"] = df.item_id.apply(lambda x: "m" + x)
        it["item_id"] = it.item_id.apply(lambda x: "m" + x)

        if args.validation:
            df2 = dataset2.train_interactions.copy()
        else:
            df2 = dataset2.full_train_interactions.copy()

        it2 = dataset2.items_texts.copy()
        df2["user_id"] = df2.user_id.apply(lambda x: "g" + x)
        df2["item_id"] = df2.item_id.apply(lambda x: "g" + x)
        it2["item_id"] = it2.item_id.apply(lambda x: "g" + x)
        _train_interactions = pd.concat([df, df2])
        items_texts = pd.concat([it, it2])
        _train_interactions["item_id"] = _train_interactions["item_id"].astype("category")
        _train_interactions["user_id"] = _train_interactions["user_id"].astype("category")
        items_d = items_texts
        items_d["asin"] = items_d.item_id
    else:
        print("Unknown dataset. List of available datsets: \n")
        for x in config.keys():
            print(x)
        print("goodlens")
        print()
        return None, None, None
    
    return dataset, evaluator, val_evaluator, _train_interactions, items_d

def load_text_model(args, items_d, dataset, _train_interactions):
    # load and preprocess text side information
    print("Preprocessing texts.")
    if args.evaluate or args.evaluate_epoch:
        am_itemids = items_d.asin.to_numpy()
        cc = np.array(dataset.all_interactions.item_id.cat.categories)
        ccdf = pd.Series(cc).to_frame()
        ccdf.columns = ["item_id"]
        amdf = pd.Series(am_itemids).to_frame().reset_index()
        amdf.columns = ["idx", "item_id"]
        am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()

        if args.dataset in config.keys():
            am_texts = items_d._text_attributes
        elif args.preproces_html:
            am_texts = items_d.fillna(0).apply(
                lambda row: f"{row.title}: {preproces_html('. '.join(eval(row.description)))}", axis=1
            )
        else:
            print("using html preprocessing")
            am_texts = items_d.fillna(0).apply(lambda row: f"{row.title}: {'. '.join(eval(row.description))}", axis=1)

        am_texts_all = am_texts.to_numpy()[am_locator]  # evaluation texts
    else:
        am_texts_all = None

    am_itemids = items_d.asin.to_numpy()
    cc = np.array(_train_interactions.item_id.cat.categories)
    ccdf = pd.Series(cc).to_frame()
    ccdf.columns = ["item_id"]
    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns = ["idx", "item_id"]
    am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()
    am_texts = items_d._text_attributes
    am_texts = am_texts.to_numpy()[am_locator]  # training texts

    # for e5 models
    if args.prefix is not None:
        print("adding prefix", args.prefix, "to all texts")
        am_texts = np.array([args.prefix + x for x in am_texts])
        print(am_texts[:10])

    # create sentence Transformer that will be trained
    print("Creating sbert")
    sbert = SentenceTransformer(args.sbert, device=DEVICE)
    if args.max_seq_length is not None:
        sbert.max_seq_length = args.max_seq_length

    # tokenize item text side information (descriptions)
    am_tokenized = sbert.tokenize(am_texts)
    if am_texts_all is None:
        am_texts_all = am_texts
        
    return am_texts_all, am_tokenized, sbert

def load_image_model(args, items_d, dataset, _train_interactions):
    image_model = images.ImageModel(args.image_model, device=DEVICE)

    tokenized_images_dict = images.read_images_into_dict(dataset.all_interactions.item_id.cat.categories, fn=image_model.tokenize, path=dataset.images_dir, suffix=dataset.images_suffix)
    tokenized_train_images = images.read_images_from_dict(_train_interactions.item_id.cat.categories, tokenized_images_dict)
    tokenized_test_images = images.read_images_from_dict(dataset.all_interactions.item_id.cat.categories, tokenized_images_dict)

    return tokenized_test_images, tokenized_train_images, image_model

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def load_audio_model(args, dataset, _train_interactions, folder):
    #prepr_config = preprocessing_config[model_type]
    prepr_config = dataset.audio_prepr_config
    pd.Series(prepr_config).to_csv(f"{folder}/preprocessing.csv")
    audio_model = audio.AudioModel(args.audio_model, DEVICE, prepr_config,
                                   args.token_pooling, args.num_spectograms,
                                   args.more_spec_pooling)
    
    tokenized_audio_dict = audio.read_audios_into_dict(dataset.all_interactions.item_id.cat.categories, fn=audio_model.tokenize, path=dataset.audio_prepr_config["audio_dir"], metadata_path=dataset.metadata_path, device = DEVICE)
    tokenized_train_audio = audio.read_audios_from_dict(_train_interactions.item_id.cat.categories, tokenized_audio_dict)
    tokenized_test_audio = audio.read_audios_from_dict(dataset.all_interactions.item_id.cat.categories, tokenized_audio_dict)
    return tokenized_test_audio, tokenized_train_audio, audio_model

def load_audio_atributes_model(args, dataset, _train_interactions, folder):
    
    prepr_config = {
        "csv_path": dataset.csv_path_end_pooling if args.pooling_time == "end" else dataset.csv_path_emb_pooling,
        "pooling_time": args.pooling_time,
        "AST_occurence_path": dataset.AST_occurence_path,
        "standardize_ess_att": args.standardize_ess_att,
        "standardize_ast_att": args.standardize_ast_att,
        "scale_AST_att": args.scale_AST_att,
        "which_attributes": args.which_attributes
    }
    
    model_type = "audio_att_model"
    pd.Series(prepr_config).to_csv(f"{folder}/preprocessing.csv")
    att_model = audio_att.AudioAtt_wrapper(DEVICE, model_type, prepr_config, _train_interactions.item_id.cat.categories, 
                                           args.num_hidden_layers, args.hidden_dimension, args.output_dimension)
    
    tok_audio_dict = att_model.processor.read_audios_into_dict(dataset.all_interactions.item_id.cat.categories)
    tokenized_train_audio = att_model.processor.read_audios_from_dict(_train_interactions.item_id.cat.categories, tok_audio_dict)
    tokenized_test_audio = att_model.processor.read_audios_from_dict(dataset.all_interactions.item_id.cat.categories, tok_audio_dict)
    return tokenized_test_audio, tokenized_train_audio, att_model

def load_audio_embeddings_model(args, dataset, _train_interactions, folder):
    
    prepr_config = {
        "embeddings_json_path": dataset.embeddings_json_path,
        "standardize": args.standardize,
    }
    
    model_type = "audio_emb_model"
    pd.Series(prepr_config).to_csv(f"{folder}/preprocessing.csv")
    att_model = audio_att.AudioAtt_wrapper(DEVICE, model_type, prepr_config, _train_interactions.item_id.cat.categories, 
                                           args.num_hidden_layers, args.hidden_dimension, args.output_dimension)
    
    tok_audio_dict = att_model.processor.read_audios_into_dict(dataset.all_interactions.item_id.cat.categories)
    tokenized_train_audio = att_model.processor.read_audios_from_dict(_train_interactions.item_id.cat.categories, tok_audio_dict)
    tokenized_test_audio = att_model.processor.read_audios_from_dict(dataset.all_interactions.item_id.cat.categories, tok_audio_dict)
    return tokenized_test_audio, tokenized_train_audio, att_model
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def prepare_schedule(args, steps_per_epoch):
    # prepare training schedule
    if args.scheduler == "CosineDecay":
        schedule = keras.optimizers.schedules.CosineDecay(
            0.0,
            steps_per_epoch * (args.decay_epochs + args.warmup_epochs),
            alpha=0.0,
            name="CosineDecay",
            warmup_target=args.warmup_lr,
            warmup_steps=steps_per_epoch * args.warmup_epochs,
        )
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
        print("Using schedule with config", schedule.get_config())
    elif args.scheduler == "LinearWarmup":
        schedule = LinearWarmup(
            warmup_steps=steps_per_epoch * args.warmup_epochs,
            decay_steps=steps_per_epoch * args.decay_epochs,
            starting_lr=args.init_lr,
            warmup_lr=args.warmup_lr,
            final_lr=args.target_lr,
        )
        print("Using schedule with config", schedule.get_config())
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
    else:
        schedule = args.lr
        epochs = args.epochs
        print("Using constant learning rate of", schedule)
    
    return schedule, epochs

def main(args):
    # prepare logging folder
    folder = os.path.join(
        "results", f"{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    vargs["cuda_or_cpu"] = DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)

    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print(f"seeds set to {args.seed}")

    # read data
    dataset, evaluator, val_evaluator, _train_interactions, items_d = load_data(args)
    
    if dataset is None:
        return

    if args.sbert is not None: 
        am_texts_all, am_tokenized, sbert = load_text_model(args, items_d, dataset, _train_interactions)
    elif args.image_model is not None:
        am_texts_all, am_tokenized, sbert = load_image_model(args, items_d, dataset, _train_interactions)
    #----------------------------------------------------------------------------------------------------------------------------------
    elif args.audio_model is not None:
        am_texts_all, am_tokenized, sbert = load_audio_model(args, dataset, _train_interactions, folder)
    
    elif args.audio_att_model:
        am_texts_all, am_tokenized, sbert = load_audio_atributes_model(args, dataset, _train_interactions, folder)
    
    elif args.audio_emb_model:
        am_texts_all, am_tokenized, sbert = load_audio_embeddings_model(args, dataset, _train_interactions, folder)
    #----------------------------------------------------------------------------------------------------------------------------------
    
    else:
        print("Dont know what to train. Please specify the --sbert argument.")

    # training in paralel on multiple gpus
    if args.devices is not None:
        print(f"Will run sbert on devices {args.devices}")
        devices_to_run = eval(args.devices)
        module_sbert = torch.nn.DataParallel(sbert, device_ids=devices_to_run, output_device=devices_to_run[0])
    else:
        module_sbert = sbert

    # create X train
    print("Creating interaction matrix for training")
    X = get_sparse_matrix_from_dataframe(_train_interactions)

    # prepare dataloader
    print("Creating dataloader")
    datal = beeformerDataset(
        X, am_tokenized, DEVICE, shuffle=True, max_output=args.max_output, batch_size=args.batch_size
    )
    steps_per_epoch = len(datal)

    print(sbert)

    # create trainable keras model
    model = NMSEbeeformer(
        tokenized_sentences=am_tokenized,
        items_idx=_train_interactions.item_id.cat.categories,
        sbert=keras.layers.TorchModuleWrapper(module_sbert),
        device=DEVICE,
        top_k=args.top_k,
        sbert_batch_size=args.sbert_batch_size,
        num_layers_to_freeze=args.num_layers_to_freeze,
    )

    # prepare lr schedule 
    schedule, epochs = prepare_schedule(args, steps_per_epoch)

    model.to(DEVICE)

    # create callback object to monitor the training procedure
    cbs = []
    if args.evaluate or args.evaluate_epoch or args.save_every_epoch:
        eval_cb = evaluateWriter(
            items_idx=dataset.all_interactions.item_id.cat.categories,
            sbert=sbert,
            evaluator=val_evaluator if args.validation else evaluator,
            logdir=folder,
            DEVICE=DEVICE,
            texts=am_texts_all,
            sbert_name=args.model_name,
            evaluate_epoch=args.evaluate_epoch,
            save_every_epoch=args.save_every_epoch,
        )
        cbs.append(eval_cb)

    # build the model
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=schedule), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()]
    )
    print("Building the model")
    model.train_step(datal[0])
    model.built = True
    model.summary()
    print("Starting training loop")
    train_time = 0

    # training
    fits = []
    print(f"Training for {args.warmup_epochs+args.decay_epochs+args.tuning_epochs} epochs.")
    f = model.fit(
        datal,
        epochs=epochs,
        callbacks=cbs,
    )
    fits.append(f)
    train_time = time.time() - train_time
       
    # save resulting model
    #sbert.save(args.model_name)
    # final evaluation
    if args.evaluate:
        embs = sbert.encode(am_texts_all, show_progress_bar=True)
        model = SparseKerasELSA(
            len(dataset.all_interactions.item_id.cat.categories),
            embs.shape[1],
            dataset.all_interactions.item_id.cat.categories,
            device=DEVICE,
        )
        model.to(DEVICE)
        model.set_weights([embs])
                
        if args.use_cold_start:
            df_preds = model.predict_df(
                evaluator.test_src,
                candidates_df=(
                    evaluator.cold_start_candidates_df if hasattr(evaluator, "cold_start_candidates_df") else None
                ),
                k=1000,
            )
            df_preds = df_preds[
                ~df_preds.set_index(["item_id", "user_id"]).index.isin(
                    evaluator.test_src.set_index(["item_id", "user_id"]).index
                )
            ]
        else:
            # if validation is used and evaluate_epoch is not set, we need to evaluate the model on the validation set  
            if args.validation and not args.evaluate_epoch:
                print("predicting on validation set")     
                df_preds = model.predict_df(val_evaluator.test_src)
                results = val_evaluator(df_preds)

                df = pd.DataFrame([results])
                df["epoch"] = epochs
                df = df[list(df.columns[-1:]) + list(df.columns[:-1])]  # Reorder columns

                save_to = folder + "/val_logs.csv"
                file_exists = os.path.isfile(save_to)
                df.to_csv(save_to, mode="a", header=not file_exists, index=False)
            
            print("predicting on testing set")
            df_preds = model.predict_df(evaluator.test_src)
            
        results = evaluator(df_preds)
        print(results)
        pd.Series(results).to_csv(f"{folder}/result.csv")
        print("results file written")

    # final logs
    ks = list(f.history.keys())
    dc = {k: np.array([(f.history[k]) for f in fits]).flatten() for k in ks}
    dc["epoch"] = np.arange(len(dc[list(dc.keys())[0]])) + 1
    df = pd.DataFrame(dc)
    df[list(df.columns[-1:]) + list(df.columns[:-1])]

    df.to_csv(f"{folder}/history.csv")
    print("history file written")

    try:
        pd.concat([pd.Series(x).to_frame().T for x in eval_cb.results_list]).to_csv(f"{folder}/results-history.csv")
    except:
        print("eval_cb not exist")

    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")

    out = subprocess.check_output(["nvidia-smi"])

    with open(os.path.join(folder, f"{args.dataset}_{args.flag}.log"), "w") as f:
        f.write(out.decode("utf-8"))


if __name__ == "__main__":
    main(args)
