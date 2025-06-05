# 🎓 Bachelor's Thesis

This project focuses on building and evaluating recommendation models using **feature extraction**, **content-based methods**, **hybrid approaches**, and **baseline comparisons**.

The main objective was to **extract meaningful attributes** and **test them** in both **content-only** and **hybrid recommendation** settings.  
The core of the system is inspired by the **beeFormer** model.

Due to data usage restrictions, the dataset employed in this thesis cannot be publicly shared. Consequently, the provided code cannot be executed in its current form.

---

## 📂 Project Structure

Here’s an overview of the key folders and files:

- **Feature Extraction**
  - Located in the `Feature_extraction/` directory.
  - Scripts for generating embeddings and attribute vectors.
- **Models**
  - Stored in the `models/` directory.
  - Contains different architectures used across experiments.

    ##### Contents of `models/` based on task:

    - **Content-Based Models**
      - Main scripts:  
        - `content_based_similarity.py`  
        - `models.py`  
        - `Attributes_processor.py`

    - **Hybrid Models**
      - Main scripts:  
        - `train.py`  
        - `models.py`  
        - `audio_att.py`  
        - `Attributes_processor.py`

    - **Audio-based BeeFormer**
      - Main scripts:  
        - `train.py`  
        - `models.py`  
        - `audio.py`  
        - `AST_processor.py`

    - **Configuration**
      - General settings for models and datasets:  
        - `config.py`
---


# 🛠️ How to Run

These commands are all run from the `models` directory, with the exception of Attribute extraction, which is run from the `Feature_extraction` directory.

## 1. ATTRIBUTE EXTRACTION

```bash
python3 Attribute_extractor_run.py --device 0 --Only_generate_embeddings True --pooling mean
```

---

## 2. SIMILARITY CHOICE

**Embeddings:**

```bash
python3 run_experiments.py --config grid_search/similarity_measures_emb.json
```

**Attributes:**

```bash
python3 run_experiments.py --config grid_search/similarity_measures_att.json
```

---

## 3. HYBRID

### Audio beeFormer
```bash
python3 train.py --audio_model bookbot/distil-ast-audioset --dataset audiodataset_audio --device 2 --epochs 1 --sbert_batch_size 6 --batch_size 1024 --max_output 2500 --evaluate true --evaluate_epoch true --validation true --num_spectograms 3 --num_layers_to_freeze 6
```

### Emb beeFormer

```bash
python3 train.py --audio_emb_model true --dataset audiodataset_audio --device 3 --epochs 10 --sbert_batch_size 256 --batch_size 1024 --max_output 5000 --save_every_epoch false --validation True --evaluate true --evaluate_epoch true --lr 5e-5
```

### Att beeFormer

```bash
python3 train.py --audio_att_model true --dataset audiodataset_audio --device 3 --epochs 10 --sbert_batch_size 256 --batch_size 1024 --max_output 1024 --save_every_epoch false --validation True --evaluate true --evaluate_epoch true
```

---

## 4. CONTENT-BASED METHODS

### Embeddings
```bash
python3 content_based_similarity.py --device 0 --validation True --audio_emb true
```

### Attributes
```bash
python3 content_based_similarity.py --device 0 --validation True --audio_att true
```
---

## 5. BASELINES

### ALS

```bash
python3 experiment_ALS.py --device 1 --dataset audiodataset_audio --pu 5 --pi 1 --use_gpu true
```

### EASE

```bash
python3 experiment_EASE.py --dataset audiodataset_audio --pu 5
```

### ELSA (Dense)

```bash
python3 experiment_ELSA.py --dataset audiodataset_audio --pu 5
```

### KNN

```bash
python3 experiment_KNN.py --pu 5 --dataset audiodataset_audio
```

### Att HEATER

```bash
python3 HEATER.py --device 3 --epochs 10 --dataset audiodataset_audio --audio_att_model true --als_use_gpu true --validation true
```

### Emb HEATER

```bash
python3 HEATER.py --device 3 --epochs 10 --dataset audiodataset_audio --audio_emb_model true --standardize true --als_use_gpu true --validation true --evaluate_epoch true
```

### TOP-POP

```bash
python3 experiment_top_pop.py --pu 5 --dataset audiodataset_audio
```

### RANDOM

```bash
python3 experiment_random.py --dataset audiodataset_audio --validation true --pu 5
```

## Acknowladgement
Tento software vznikl za podpory Fakulty informačních technologií ČVUT v Praze, fit.cvut.cz

![Logo FIT](logo-FIT.jpg)

