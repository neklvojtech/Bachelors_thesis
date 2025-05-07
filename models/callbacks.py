# =============================================================================
# File: callbacks.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 26.3.2025
# Description: Defines callbacks for Keras models to evaluate and log results during training.
# Notes: Modified as part of the bachelor's thesis work. Heater callbacks were added.
# =============================================================================


import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import pandas as pd

from models import SparseKerasELSA
from time import time
from utils import *


class evaluateWriter(keras.callbacks.Callback):
    def __init__(
        self,
        items_idx,
        sbert,
        texts,
        evaluator,
        logdir,
        DEVICE,
        sbert_name="sbert_temp_model",
        evaluate_epoch="false",
        save_every_epoch="false",
    ):
        super().__init__()
        self.evaluator = evaluator
        self.logdir = logdir
        self.sbert = sbert
        self.texts = texts
        self.items_idx = items_idx
        self.DEVICE = DEVICE
        self.results_list = []
        self.sbert_name = sbert_name
        self.evaluate_epoch = evaluate_epoch
        self.save_every_epoch = save_every_epoch

    def on_epoch_end(self, epoch, logs=None):
        print()
        if self.save_every_epoch:
            print("saving sbert model")
            self.sbert.save(f"{self.sbert_name}-epoch-{epoch}")
        if self.evaluate_epoch:
            embs = self.sbert.encode(self.texts, show_progress_bar=True)
            model = SparseKerasELSA(len(self.items_idx), embs.shape[1], self.items_idx, device=self.DEVICE)
            model.to(self.DEVICE)
            model.set_weights([embs])
            if isinstance(self.evaluator, ColdStartEvaluation):
                df_preds = model.predict_df(
                    self.evaluator.test_src,
                    candidates_df=(
                        self.evaluator.cold_start_candidates_df
                        if hasattr(self.evaluator, "cold_start_candidates_df")
                        else None
                    ),
                    k=1000,
                )
                df_preds = df_preds[
                    ~df_preds.set_index(["item_id", "user_id"]).index.isin(
                        self.evaluator.test_src.set_index(["item_id", "user_id"]).index
                    )
                ]
            else:
                df_preds = model.predict_df(self.evaluator.test_src)
                
            results = self.evaluator(df_preds)

            df = pd.DataFrame([results])
            df["epoch"] = epoch + 1
            df = df[list(df.columns[-1:])+list(df.columns[:-1])]

            # Append to CSV file
            save_to = self.logdir + "/val_logs.csv"
            file_exists = os.path.isfile(save_to)
            df.to_csv(save_to, mode="a", header=not file_exists, index=False)
            self.results_list.append(results)
            
class Callbacks_HEATER(keras.callbacks.Callback):
    def __init__(
        self,
        als_model,
        evaluator,
        logdir,
        DEVICE,
        validation,
        evaluate_epoch,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.logdir = logdir
        self.als_model = als_model
        self.DEVICE = DEVICE
        self.results_list = []
        self.validation = validation
        self.evaluate_epoch = evaluate_epoch
        
    def on_epoch_end(self, epoch, logs=None):
        if self.evaluate_epoch:
            print()
            df_preds = self.model.predict_df(self.evaluator.test_src, als_model=self.als_model)
            results = self.evaluator(df_preds)

            df = pd.DataFrame([results])
            df["epoch"] = epoch + 1
            df = df[list(df.columns[-1:]) + list(df.columns[:-1])]  # Reorder columns

            write_to = "/val_logs.csv" if self.validation else "/training_logs.csv"  
            save_to = self.logdir + write_to
            file_exists = os.path.isfile(save_to)
            df.to_csv(save_to, mode="a", header=not file_exists, index=False)
            
            # Store the results for tracking
            self.results_list.append(results)