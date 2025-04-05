# Project Summary

This repository contains solutions for three related projects in high-energy physics event analysis:

1. **Common Task: Electron/Photon Classification with ResNet-15**  
2. **Specific Task 2A: Event Classification with Transformer Autoencoders**  
3. **Specific Task 2H: Jet Classification Using Graph Neural Networks**

Each project leverages a different deep learning architecture tailored to the underlying physics data, demonstrating effective classification performance while addressing domain-specific challenges.

---

## 1. Common Task: Electron/Photon Classification with ResNet-15

### Overview
The goal is to classify two types of particles—electrons and photons—using image data. Each sample is a 32×32 matrix with two channels:
- **Channel 1:** Hit Energy
- **Channel 2:** Time

A ResNet-15–like architecture is employed to achieve high classification performance while ensuring robust generalization. The model is trained on 80% of the data and evaluated on the remaining 20%.

### Dataset
- **Data Files:** Two HDF5 files (one for electrons and one for photons)
- **Samples per File:** 249,000 images
- **Image Dimensions:** 32×32 with 2 channels

### Results
- **Test Loss:** 0.5512  
- **Test Accuracy:** 72.89%  
- **ROC AUC:** 0.80

![Electron/Photon Classification Result](https://github.com/user-attachments/assets/41b7158b-8ba8-47c7-b3d4-8c5af9509bd3)

---

## 2. Specific Task 2A: Event Classification with Masked Transformer Autoencoders

### Problem Overview
This task focuses on event classification by learning meaningful representations from high-energy physics events. A masked Transformer autoencoder is used to pretrain on event data; its latent space is then leveraged by a downstream classifier to discriminate between signal and background events.

### Dataset
- **Source:** HIGGS dataset from the UCI Machine Learning Repository
- **Data Subset:**
  - **Training/Pretraining:** First 1.1 million events (using only the first 21 features)
  - **Testing:** Last 100,000 events
- **Feature Details:**  
  Kinematic properties including transverse momentum (pT), pseudorapidity (η), and azimuthal angle (φ). The φ features are transformed into cosine and sine components for improved representation.

### Approach
1. **Data Preprocessing:**
   - **Feature Selection:** Use only the first 21 features.
   - **Angular Transformation:** Convert φ features into cosine and sine representations.
   - **Standardization:** Scale data to achieve uniform distribution.

2. **Autoencoder Pretraining:**
   - **Architecture:**  
     A Transformer-based autoencoder with an encoder that embeds the input features (with positional encodings) and a decoder that reconstructs the input via a learnable target embedding.
   - **Custom Loss Components:**  
     In addition to standard reconstruction loss, the training incorporates:
     - Momentum Conservation Loss
     - Invariant Mass Loss
     - Angular (φ) Loss
     - Transverse Momentum Loss
     - Pseudorapidity (η) Loss

3. **Classifier Development:**
   - **Pretrained Encoder:** The encoder is reused (parameters frozen).
   - **Classifier:** A fully connected network on top of the encoder’s latent space to perform binary classification (signal vs. background).
   - **Evaluation:** Performance is measured using ROC-AUC.

### Autoencoder Performance Metrics

| Feature      | MSE    | MAE    | Pearson Corr. |
|--------------|--------|--------|---------------|
| lepton_pt    | 0.0921 | 0.2143 | 0.8755        |
| lepton_eta   | 0.0011 | 0.0204 | 0.9996        |
| lepton_phi   | 0.4066 | 0.4374 | 0.9473        |
| met          | 0.1052 | 0.2277 | 0.8768        |
| met_phi      | 0.4693 | 0.4625 | 0.9351        |
| jet1_pt      | 0.0431 | 0.1556 | 0.9152        |
| jet1_eta     | 0.0023 | 0.0336 | 0.9990        |
| jet1_phi     | 0.2370 | 0.3885 | 0.9824        |
| jet1_btag    | 0.0018 | 0.0363 | 0.9999        |
| jet2_pt      | 0.0663 | 0.1848 | 0.8840        |
| jet2_eta     | 0.0027 | 0.0393 | 0.9990        |
| jet2_phi     | 0.2456 | 0.3704 | 0.9734        |
| jet2_btag    | 0.0029 | 0.0457 | 0.9999        |
| jet3_pt      | 0.0717 | 0.1981 | 0.8764        |
| jet3_eta     | 0.0018 | 0.0225 | 0.9993        |
| jet3_phi     | 0.3502 | 0.4216 | 0.9583        |
| jet3_btag    | 0.0036 | 0.0537 | 0.9999        |
| jet4_pt      | 0.0785 | 0.2053 | 0.8740        |
| jet4_eta     | 0.0015 | 0.0212 | 0.9996        |
| jet4_phi     | 0.3334 | 0.3939 | 0.9563        |
| jet4_btag    | 0.0047 | 0.0515 | 0.9999        |

### Classification Performance

- **Validation ROC-AUC:** ~0.73  
- **Training Progress:** Both loss and AUC metrics improved consistently over epochs, with the ROC curve demonstrating effective class discrimination.

![ROC Curve](https://github.com/user-attachments/assets/621035c4-045f-4d6c-a8e4-91a5396da075)  
![Training AUC](https://github.com/user-attachments/assets/fdf1f18b-1235-45d0-aa76-1bea8b0d35f0)

### Summary
The integration of physics-informed loss functions within a Transformer autoencoder yields robust latent representations that facilitate effective event classification. The final classifier exhibits strong separation between signal and background events.

---

## 3. Specific Task 2H: Jet Classification Using Graph Neural Networks

### Problem Overview
This task tackles jet classification by converting jet images into point cloud graphs and applying graph neural network (GNN) models for classification. Two architectures are explored:
- **Graph Edge Model:** Utilizes EdgeConv layers.
- **Graph Attention Model:** Utilizes GATConv layers with graph normalization.

### Data Processing
- **Data Source:** Parquet file containing jet images.
- **Conversion Process:**  
  - **Image to Point Cloud:** Nonzero pixels are extracted along with their features.
  - **Graph Construction:** k-nearest neighbor (kNN) is used to build graph connectivity, with edges weighted by a Gaussian kernel. Node features include pixel intensity and normalized spatial coordinates.

### Models and Training

#### Graph Edge Model
- **Architecture:**  
  Three EdgeConv layers followed by ReLU activations, global mean and max pooling, and two fully connected layers.
- **Training Log (Epoch 030):**

| Epoch | Train Loss | Train Accuracy | Train AUC |
|-------|------------|----------------|-----------|
| 030   | 0.5804     | 70.17%         | 0.7787    |
---
![image](https://github.com/user-attachments/assets/6cbe75f7-449f-4d79-8795-0e2fe1746398)

#### Graph Attention Model
- **Architecture:**  
  Three GATConv layers with graph normalization, followed by global pooling and two fully connected layers.
- **Training Log (Epoch 030):**

| Epoch | Train Loss | Train Accuracy | Train AUC |
|-------|------------|----------------|-----------|
| 030   | 0.6263     | 66.01%         | 0.7815    |
---
![image](https://github.com/user-attachments/assets/1fe9b7df-73cb-40c5-b446-be6168968501)

### Evaluation
- **ROC Curves:**  
  ROC curves generated on the test set show both models achieving ROC-AUC scores around 0.76, demonstrating effective class separation.


### Summary
Graph-based approaches successfully capture the spatial and intensity features of jet images. The Graph Edge Model achieves slightly better loss and accuracy, while the Graph Attention Model shows comparable AUC performance. Both models highlight the potential of GNNs for jet classification tasks in high-energy physics.
