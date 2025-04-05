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
  
- **Classification Performance:**  
  The final classifier reached a promising ROC-AUC score (e.g., approximately 0.73 on validation), indicating robust separation between event classes. Loss and AUC metrics improved consistently over training epochs, and the ROC curve demonstrated effective discriminative performance.

Overall, the approach successfully integrates physical principles within a modern Transformer framework, leading to meaningful event representations and effective event classification.


