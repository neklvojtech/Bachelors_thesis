# =============================================================================
# File: models.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 7.2.2025
# Description: Implements custom models.
# Notes: Modified as part of the bachelor's thesis work. Similarity model was added.
# =============================================================================

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import math
import numpy as np
import torch

from dataloaders import *
from layers import *
from utils import get_first_item

# beeformer optimized with nmse (expected loss since all the normalizations inside the train step)
class NMSEbeeformer(keras.models.Model):
    def __init__(self, tokenized_sentences, items_idx, sbert, device, top_k=0, sbert_batch_size=128, num_layers_to_freeze = None):
        super().__init__()
        self.device = device
        self.sbert = LayerSBERT(sbert, device, tokenized_sentences)
        if num_layers_to_freeze is not None:
            self.sbert.freeze_parameters(num_layers_to_freeze)
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.top_k = top_k
        self.sbert_batch_size = sbert_batch_size

    def call(self, x):
        return self.sbert(x)

    def train_step(self, data):
        # Unpack the data
        a, b = data
        x, y = a
        y = torch.hstack((x, y))
        x_out = y
        tokenized_items, slicer, negative_slicer = b
        slicer = slicer.to(self.device)
        if negative_slicer is not None:
            negative_slicer = negative_slicer.to(self.device)

        # init everything for training
        self.zero_grad()
        sbert_batch_size = self.sbert_batch_size
        len_sentences = get_first_item(tokenized_items).shape[0]
        max_i = math.ceil(len_sentences / sbert_batch_size)
        
        # sbert forward pass #1 - we want to get embeddings for items to compute loss
        with torch.no_grad():
            # we are doing it in batches because of memory
            batched_results = []
            for i in range(max_i):
                ind = i * sbert_batch_size
                ind_min = ind
                ind_max = ind + sbert_batch_size
                batch_result = self.sbert({k: v[ind_min:ind_max] for k, v in tokenized_items.items()})
                batched_results.append(batch_result)
            A = torch.vstack(batched_results)

        # track gradients for A, this will be our gradient checkpoint
        A.requires_grad = True

        # compute ELSA forward pass only for rows with values
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)
        A_negative_slicer = A[negative_slicer]
        A_negative_slicer = torch.nn.functional.normalize(A_negative_slicer, dim=-1)
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)

        # ELSA step
        xA = torch.matmul(x, A_slicer)
        xAAT = torch.matmul(xA, A_negative_slicer.T)
        y_pred = keras.activations.relu(xAAT - x_out)

        # theoretically, this might improve performance for bigger dataset
        if self.top_k > 0:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        # compute loss
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # compute gradients for the gradient checkpoint (our ELSA A matrix)
        loss.backward()

        # sbert forward pass #2
        # now we will do the sbert forward pass again, but this time we will track gradients this time, for memory reasons in again batches
        batched_results = []
        for i in range(max_i):
            ind = i * sbert_batch_size
            ind_min = ind
            ind_max = ind + sbert_batch_size
            # actual forward pass
            temp_out = self.sbert({k: v[ind_min:ind_max] for k, v in tokenized_items.items()})
            # we need to get gradients for part of A
            temp_out.retain_grad()
            # get the slice of corresponding gradients
            partial_A_grad = A.grad[ind_min:ind_max]
            # compute gradients for sbert
            temp_out.backward(gradient=partial_A_grad)

        # get gradients for sbert
        trainable_weights = [v for v in self.sbert.trainable_weights]
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


# ELSA model optimized for sparse data, used only for predictions
class SparseKerasELSA(keras.models.Model):
    def __init__(self, n_items, n_dims, items_idx, device, top_k=0):
        super().__init__()
        self.device = device
        self.ELSA = LayerELSA(n_items, n_dims, device=device)
        self.items_idx = items_idx
        self.ELSA.build()
        self(np.zeros([1, n_items]))
        self.finetuning = False
        self.top_k = top_k

    def call(self, x):
        return self.ELSA(x)

    def train_step(self, data):
        # Unpack the data
        if len(data) == 2:
            full_x = None
            a, b = data
            x, y = a
            y = torch.hstack((x, y))
            slicer, negative_slicer = b

        elif len(data) == 3:
            full_x, slicer, negative_slicer = data
        else:
            full_x, slicer = data
            negative_slicer = None

        if full_x is not None:
            if negative_slicer is not None:
                y = full_x[:, negative_slicer]
            else:
                y = full_x

            x = full_x[:, slicer]

            x = x.to(self.device)
            y = y.to(self.device)

        x = torch.nn.functional.normalize(x, p=1.0, dim=-1)
        y = torch.nn.functional.normalize(y, p=1.0, dim=-1)

        x_out = y

        self.zero_grad()

        A = self.ELSA.A
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(A_slicer, dim=-1)

        if negative_slicer is not None:
            A_negative_slicer = A[negative_slicer]
            A_negative_slicer = torch.nn.functional.normalize(A_negative_slicer, dim=-1)
        else:
            A_negative_slicer = torch.nn.functional.normalize(A, dim=-1)

        xA = torch.matmul(x, A_slicer)
        xAAT = torch.matmul(xA, A_negative_slicer.T)
        y_pred = xAAT - x_out

        if self.finetuning:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

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
        return {m.name: m.result() for m in self.metrics}

    def predict_df(self, df, k=100, user_ids=None, candidates_df=None, block_reminder=True):
        # create predictions from data in dataframe, returns predictions in dataframe
        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(candidates_df, item_indices=self.items_idx).toarray()
            candidates_vec = torch.from_numpy(candidates_vec)  # .to(self.device)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024)

        dfs = []

        for i in tqdm(range(len(data)), total=len(data), desc="ELSA evaluation"):
            x, batch_uids = data[i]

            batch = torch.from_numpy(self.predict_on_batch(x))
            if block_reminder:
                mask = 1 - x.astype(bool)  # block reminder
                batch = batch * mask

            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            df = pd.DataFrame(
                {
                    "user_id": np.stack([batch_uids] * k).flatten("F"),
                    "item_id": np.array(self.items_idx)[indices_].flatten(),
                    "value": values_.flatten(),
                }
            )
            df["user_id"] = df["user_id"].astype(str).astype("category")
            df["item_id"] = df["item_id"].astype(str).astype("category")
            dfs.append(df)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype("category")
        df["item_id"] = df["item_id"].astype(str).astype("category")
        return df
    
class KerasELSA(keras.models.Model):
    def __init__(self, n_items, n_dims, items_idx, device):
        super().__init__()
        #self.A = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])))
        self.device = device
        self.ELSA = LayerELSA(n_items, n_dims, device=device)
        self.items_idx = items_idx
        self.ELSA.build()

    def call(self, x):
        return self.ELSA(x)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        y_pred = self(x, training=True)  # Forward pass
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
        #print({m.name: m.result() for m in self.metrics})
        return {m.name: m.result() for m in self.metrics}

    def predict_sparse(self, x):
        data = BasicRecSysDataset(x)
        return self.predict(data)

    def predict_df(self, df, k=100, user_ids=None):

        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        #x = get_sparse_matrix_from_dataframe(df, item_indices=self.items_idx)

        data = PredictDfRecSysDataset(df, self.items_idx)

        dfs = []
        imin = 0
        auser_ids = user_ids

        for i in tqdm(range(len(data)), total=len(data)):
            x, batch_uids = data[i]

            batch = torch.from_numpy(self.predict_on_batch(x))
            mask = 1-x.astype(bool)  # block reminder
            batch = batch * mask

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            df = pd.DataFrame({"user_id": np.stack([batch_uids]*k).flatten("F"), "item_id": np.array(
                self.items_idx)[indices_].flatten(), "value": values_.flatten()})
            df["user_id"] = df["user_id"].astype(str).astype('category')
            df["item_id"] = df["item_id"].astype(str).astype('category')
            dfs.append(df)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype('category')
        df["item_id"] = df["item_id"].astype(str).astype('category')
        return df


class EmbeddingSimilarityModel:
    def __init__(self, item_embeddings, items_idx, similarity="cosine", device="cpu"):
        """
        Initializes the EmbeddingSimilarityModel with given item embeddings, item indices, 
        similarity metric, and the device for computation.

        Args:
            item_embeddings (torch.Tensor): The embeddings for the items, shape [num_items x dim].
            items_idx (list or np.array): A list or array of item ids used for mapping back to the DataFrame.
            similarity (str, optional): The type of similarity metric to use. Can be "cosine", "dot", or "euclidean". Default is "cosine".
            device (str, optional): The device to use for computations, default is "cpu".

        """
        self.A = item_embeddings.float().to(device)
        self.device = device
        self.similarity = similarity
        self.items_idx = items_idx

    def _similarity(self, x):
        """
        x: interaction vector (batch_size x num_items)
        returns similarity matrix (batch_size x num_items)
        """
        A = self.A

        # User embeddings from x @ A
        user_embs = torch.matmul(x, A)
        item_embs = A

        if self.similarity == "cosine":
            A = torch.nn.functional.normalize(self.A, dim=-1)
            xA = torch.matmul(x, A)
            xAAT = torch.matmul(xA, A.T)
            return xAAT

        elif self.similarity == "dot":
            return torch.matmul(user_embs, item_embs.T)

        elif self.similarity == "euclidean":
            user_sq = (user_embs**2).sum(dim=1).unsqueeze(1)
            item_sq = (item_embs**2).sum(dim=1).unsqueeze(0)
            dot_product = torch.matmul(user_embs, item_embs.T)
            dists = user_sq + item_sq - 2 * dot_product
            return -dists  # negative distance = similarity

        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

    def predict_df(self, df, k=100, candidates_df=None, block_reminder=True):

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(candidates_df, item_indices=self.items_idx).toarray()
            candidates_vec = torch.from_numpy(candidates_vec).to(self.device)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=1024)

        dfs = []

        for i in tqdm(range(len(data)), desc="SimilarityModel eval"):
            x, batch_uids = data[i]
            x = torch.from_numpy(x).float().to(self.device)

            batch = self._similarity(x)

            if block_reminder:
                mask = (~x.bool()).float()
                batch = batch * mask

            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            df = pd.DataFrame({
                "user_id": np.stack([batch_uids] * k).flatten("F"),
                "item_id": np.array(self.items_idx)[indices_].flatten(),
                "value": values_.flatten(),
            })
            df["user_id"] = df["user_id"].astype(str).astype("category")
            df["item_id"] = df["item_id"].astype(str).astype("category")
            dfs.append(df)

        df = pd.concat(dfs)
        df["user_id"] = df["user_id"].astype(str).astype("category")
        df["item_id"] = df["item_id"].astype(str).astype("category")
        return df