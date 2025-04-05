# Common Task
## Electron/Photon Classification with ResNet-15

This project demonstrates a solution for classifying electrons and photons using a ResNet-15–like architecture. The classification is performed on a dataset of 32×32 images with two channels (hit energy and time) corresponding to two types of particles (electrons and photons) detected in a high-energy physics experiment.


### Overview

In this task, the goal is to design a model that can classify two types of particles—electrons and photons—using image data where each sample is a 32×32 matrix with two channels:
- **Channel 1:** Hit energy
- **Channel 2:** Time

A ResNet-15 architecture is employed to achieve high classification performance while ensuring that the model does not overfit to the test dataset. The final model is trained on 80% of the data and evaluated on the remaining 20%.

---

### Dataset

The dataset consists of two HDF5 files:
- **Photons:**
- **Electrons:**
Each file contains 249,000 samples. Each sample is a 32×32 image with 2 channels.


