# Common Task
## Electron/Photon Classification with ResNet-15

This project demonstrates a solution for classifying electrons and photons using a ResNet-15–like architecture. The classification is performed on a dataset of 32×32 images with two channels (hit energy and time) corresponding to two types of particles (electrons and photons) detected in a high-energy physics experiment.


### Overview

In this task, the goal is to design a model that can classify two types of particles—electrons and photons—using image data where each sample is a 32×32 matrix with two channels:
- **Channel 1:** Hit energy
- **Channel 2:** Time

![image](https://github.com/user-attachments/assets/26a6f068-1465-4825-8dc5-563a0973cd5e)

A ResNet-15 like architecture is employed to achieve high classification performance while ensuring that the model does not overfit to the test dataset. The final model is trained on 80% of the data and evaluated on the remaining 20%.

---

### Dataset

The dataset consists of two HDF5 files:
- **Photons:**
- **Electrons:**
Each file contains 249,000 samples. Each sample is a 32×32 image with 2 channels.

### Results
![image](https://github.com/user-attachments/assets/41b7158b-8ba8-47c7-b3d4-8c5af9509bd3)
---
Test Loss: 0.5512 | Test Accuracy: 0.7289 | ROC: 0.80

# Specific Task-2A
# CMS Projects: Event Classification with Masked Transformer Autoencoders

## Problem Overview
This project addresses the event classification task using a masked Transformer autoencoder. The goal is to learn meaningful representations of high-energy physics events and utilize these features to classify events as signal or background. The approach leverages a Transformer-based autoencoder to pretrain on event data and then fine-tunes a classifier on top of the learned latent space.

## Dataset
- **Source:** HIGGS dataset from the UCI Machine Learning Repository.
- **Subset:** Only the first 21 features are used, considering the first 1.1 million events for training/pretraining and the last 100k events for testing.
- **Features:** Includes kinematic properties such as transverse momentum (pT), pseudorapidity (η), and azimuthal angle (φ), with phi features transformed into cosine and sine components for better representation.

## Approach
1. **Data Preprocessing:**  
   - Selected only the required features.
   - Transformed angular features into cosine and sine representations.
   - Standardized the data to ensure uniform scale.
   
2. **Autoencoder Pretraining:**  
   - A Transformer autoencoder was designed to capture the underlying structure of the event data.
   - Custom loss functions were incorporated to enforce physical constraints (e.g., momentum conservation, invariant mass consistency) along with the reconstruction loss.
   - The autoencoder was trained on the training set with validation to ensure generalizability.

3. **Classifier Development:**  
   - The pretrained encoder from the autoencoder was reused, with its parameters frozen.
   - A classifier network was built on top of the encoder's latent representations.
   - The classifier was trained to predict the event class, with its performance evaluated using ROC-AUC.

## Results
- **Autoencoder Performance:**  
  The autoencoder achieved low reconstruction error while maintaining physical consistency, as evidenced by low error metrics (MSE and MAE) across key features and high correlation between true and reconstructed physics quantities.
  ![image](https://github.com/user-attachments/assets/d814628c-a610-4e5e-b993-2cc39e817e04)
  Feature-wise Reconstruction Metrics:
lepton_pt: MSE=0.0921, MAE=0.2143, Pearson=0.8755
lepton_eta: MSE=0.0011, MAE=0.0204, Pearson=0.9996
lepton_phi: MSE=0.4066, MAE=0.4374, Pearson=0.9473
met: MSE=0.1052, MAE=0.2277, Pearson=0.8768
met_phi: MSE=0.4693, MAE=0.4625, Pearson=0.9351
jet1_pt: MSE=0.0431, MAE=0.1556, Pearson=0.9152
jet1_eta: MSE=0.0023, MAE=0.0336, Pearson=0.9990
jet1_phi: MSE=0.2370, MAE=0.3885, Pearson=0.9824
jet1_btag: MSE=0.0018, MAE=0.0363, Pearson=0.9999
jet2_pt: MSE=0.0663, MAE=0.1848, Pearson=0.8840
jet2_eta: MSE=0.0027, MAE=0.0393, Pearson=0.9990
jet2_phi: MSE=0.2456, MAE=0.3704, Pearson=0.9734
jet2_btag: MSE=0.0029, MAE=0.0457, Pearson=0.9999
jet3_pt: MSE=0.0717, MAE=0.1981, Pearson=0.8764
jet3_eta: MSE=0.0018, MAE=0.0225, Pearson=0.9993
jet3_phi: MSE=0.3502, MAE=0.4216, Pearson=0.9583
jet3_btag: MSE=0.0036, MAE=0.0537, Pearson=0.9999
jet4_pt: MSE=0.0785, MAE=0.2053, Pearson=0.8740
jet4_eta: MSE=0.0015, MAE=0.0212, Pearson=0.9996
jet4_phi: MSE=0.3334, MAE=0.3939, Pearson=0.9563
jet4_btag: MSE=0.0047, MAE=0.0515, Pearson=0.9999

  
- **Classification Performance:**  
  The final classifier reached a promising ROC-AUC score (e.g., approximately 0.73 on validation), indicating robust separation between event classes. Loss and AUC metrics improved consistently over training epochs, and the ROC curve demonstrated effective discriminative performance.

![image](https://github.com/user-attachments/assets/621035c4-045f-4d6c-a8e4-91a5396da075)
![image](https://github.com/user-attachments/assets/fdf1f18b-1235-45d0-aa76-1bea8b0d35f0)


Overall, the approach successfully integrates physical principles within a modern Transformer framework, leading to meaningful event representations and effective event classification.


