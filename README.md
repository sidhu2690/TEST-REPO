# Project Summary

This repository contains two related projects in high-energy physics event analysis:

1. **Electron/Photon Classification with ResNet-15**  
2. **Event Classification with Masked Transformer Autoencoders (Specific Task 2A)**

---

## Common Task: Electron/Photon Classification with ResNet-15

### Overview
The goal is to classify two types of particles—electrons and photons—using image data. Each sample is a 32×32 matrix with two channels:
- **Channel 1:** Hit Energy
- **Channel 2:** Time

A ResNet-15–like architecture is employed to achieve high classification performance while ensuring robust generalization (model is trained on 80% of the data and evaluated on the remaining 20%).

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

## Specific Task 2A: CMS Projects – Event Classification with Masked Transformer Autoencoders

### Problem Overview
This task focuses on event classification by learning meaningful representations from high-energy physics events. A masked Transformer autoencoder is used to pretrain on event data and then its latent space is leveraged for a downstream classifier that discriminates between signal and background events.

### Dataset
- **Source:** HIGGS dataset from the UCI Machine Learning Repository
- **Data Subset:**
  - **Training/Pretraining:** First 1.1 million events using only the first 21 features
  - **Testing:** Last 100,000 events
- **Feature Details:**  
  Kinematic properties including transverse momentum (pT), pseudorapidity (η), and azimuthal angle (φ). The φ features are transformed into cosine and sine components for improved representation.

### Approach
1. **Data Preprocessing:**
   - **Feature Selection:** Only the first 21 features are used.
   - **Angular Transformation:** φ features are converted to cosine and sine representations.
   - **Standardization:** Data is scaled to ensure a uniform distribution across features.

2. **Autoencoder Pretraining:**
   - **Architecture:**  
     A Transformer-based autoencoder is designed with:
     - **Encoder:** Embeds the input features and incorporates positional encodings before processing them with Transformer encoder layers.
     - **Decoder:** Uses a learnable target embedding along with positional encodings to reconstruct the original data from the latent representation.
   - **Custom Loss Components:**  
     In addition to the reconstruction loss, several physics-inspired losses are included:
     - **Momentum Conservation Loss**
     - **Invariant Mass Loss**
     - **Angular (φ) Loss**
     - **Transverse Momentum Loss**
     - **Pseudorapidity (η) Loss**

3. **Classifier Development:**
   - **Pretrained Encoder:** The encoder from the autoencoder is reused, with its parameters frozen.
   - **Classifier:** A fully connected network is built on top of the encoder’s latent representation to perform binary classification (signal vs. background).
   - **Evaluation:** Classifier performance is measured using ROC-AUC.

### Autoencoder Performance Metrics

The following table summarizes the feature-wise reconstruction metrics for the autoencoder:

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

The classifier built on the pretrained encoder achieved promising results:

- **Validation ROC-AUC:** Approximately 0.73
- **Training Progress:** Both loss and AUC metrics improved consistently over the epochs, with the ROC curve indicating effective class discrimination.

![ROC Curve](https://github.com/user-attachments/assets/621035c4-045f-4d6c-a8e4-91a5396da075)  
![Training AUC](https://github.com/user-attachments/assets/fdf1f18b-1235-45d0-aa76-1bea8b0d35f0)

### Summary
The combined approach demonstrates that integrating physics-informed loss functions within a Transformer-based autoencoder can lead to meaningful representations of event data. When these representations are leveraged for classification, the final model shows robust performance in separating signal and background events.

Overall, both projects illustrate effective use of deep learning architectures tailored for high-energy physics data analysis.
