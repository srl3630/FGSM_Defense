# Defending Against FGSM: A Comparative Analysis of Image Reconstruction Defenses

---

## Introduction

Neural Network based classification models are vulnerable to adversarial attacks, and even imperceptible changes to the human eye can drastically throw off predictions. 

This tutorial utilizes the German Traffic Sign Recognition Benchmark (GTSRB)[\[2\]](#2) to explore the Fast Gradient Sign Method (FGSM)[\[1\]](#1). We evaluate three different defense mechanisms aimed at removing these adversarial perturbations before they reach the classifier::

1.  **Linear Blur** (Baseline)
2.  **Standard Autoencoder**
3.  **U-Net Autoencoder**

> **Note:** This repository contains a fully runnable Jupyter Notebook designed for **Google Colab** (L4 GPU instance recommended). There are no prerequisites to run the code; it is fully self-contained.

## Related Work & Motivation

The concept of using image reconstruction as a filter for adversarial noise is a well-established area of research. This project builds upon two prominent defensive philosophies:

    Manifold Projection[\[3\]](#3): Research such as MagNet proposed the use of autoencoders to defend against adversarial examples. MagNet operates on the principle that adversarial perturbations push images away from the manifold of natural data. By passing them through a trained autoencoder, the images are projected back toward a clean state before being classified.  

    Guided Denoising[\[4\]](#4): More complex architectures, such as the High-Level Representation Guided Denoiser (HGD), have shown that denoising networks can be optimized not just for pixel-level accuracy, but to preserve the features that aid the classifier. This motivates our use of the U-Net architecture, which employs skip connections to maintain high-resolution spatial details that standard bottleneck autoencoders might discard.  

By comparing a simple Linear Blur against these neural-based reconstruction methods, we can quantitatively measure the "Recovery Rate"  of the system's operational safety.

---

## Project Objectives

This project quantitatively evaluates architectural defenses against adversarial attacks. The following objectives were achieved in the attached notebook:

* Implement an **FGSM** attack generator to fool a target classifier (A CNN in this case).
* Develop three distinct image preprocessing defenses.
* Evaluate the robustness of each defense using the following quantitative metrics:
    * **Accuracy**
    * **F1 Score (Macro)**
    * **Recall**
    * **Precision**
    * **Recovery Rate**([Custom Metric, Detailed below](#results--evaluation))

---

## Prerequisites & Colab Setup

This notebook is designed to be plug-and-play on Google Colab.

### 1. Environment Setup
1.  Open the `Computer_Vision_Project.ipynb` in Google Colab.
2.  Go to **Runtime** -> **Change runtime type**.
3.  Select **GPU** as the hardware accelerator (an L4 GPU was used for development and is highly recommended for training the Autoencoders).

### 2. Kaggle Dataset Download
The notebook automatically downloads the required **GTSRB - German Traffic Sign Recognition Benchmark** dataset directly from Kaggle using the `kagglehub` library. No manual API setup or credentials are required.

```python
# Snippet from the notebook: Kaggle Setup
import kagglehub

# Download GTSRB database from kaggle
DATA_PATH = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("Path to dataset files:", DATA_PATH)
```

---

## Methodology

### 1. The Attack: Fast Gradient Sign Method (FGSM)
FGSM works by utilizing the gradients of the neural network to create an adversarial image. For an input image, the method uses the gradients of the loss with respect to the input image to create a new image that maximizes the loss. [[Ref]](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

Where $\epsilon$ controls the strength of the perturbation (image noise).

### 2. The Defenses
Instead of retraining the classifier, this project uses input reconstruction to "clean" the adversarial images before they reach the classifier.

#### Defense A: Linear Blur (Baseline)
A simple Gaussian blur applied to the images using OpenCV. The hypothesis is that high-frequency adversarial noise might be smoothed out, though at the cost of losing legitimate high-frequency edge features. This is used as a baseline to compare against NN based defenses.

#### Defense B: Standard Autoencoder
A convolutional autoencoder containing an encoder (to compress the image into a latent space) and a decoder (to reconstruct the image). It is trained on clean images, teaching it to project noisy/adversarial images back to a clean state.

#### Defense C: U-Net Autoencoder
An autoencoder featuring skip connections between the encoding and decoding layers. These connections help preserve high-resolution details that a standard bottleneck autoencoder might lose, theoretically providing a much cleaner reconstruction. This is especially important given the nature of an FGSM attack, which exploits subtle perturbations to induce a misclassification. Since these attacks often hide in the fine textures of the image, the skip connections provide a 'spatial anchor' that prevents the autoencoder from inadvertently discarding the aspects of the image that aid in classification while attempting to filter out the sabotage.

---

## Results & Evaluation

To assess the performance of our defense methods, we employ a suite of standard classification metrics alongside a custom recovery calculation. 

* **Sabotage Level ($\epsilon$):** Represents the magnitude of the adversarial perturbation applied to the clean image. It is the maximum change allowed for any individual pixel, where $\epsilon = 0.0$ represents a clean baseline and higher values represent more aggressive perturbations.
* **Accuracy:** The ratio of correctly predicted traffic signs to the total number of test samples. 
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Precision (Macro):** Measures the model's reliability by calculating the average ratio of true positive predictions to the total number of positive predictions across all 43 classes. This prevents common signs (like 'Speed Limit 30') from overshadowing rare, critical signs.
    $$\text{Precision} = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FP_i}$$
* **Recall (Macro):** Measures the model's sensitivity by calculating the average ratio of true positives to the total number of actual instances for each class. 
    $$\text{Recall} = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FN_i}$$
* **F1-Score (Macro):** The harmonic mean of Precision and Recall. This is our primary metric as it weights the disparate class evenly.
    $$\text{F1} = \frac{1}{N} \sum_{i=1}^{N} 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$
* **Recovery Rate (%):** A custom metric designed to quantify the effectiveness of a defense relative to the impact of the attack. It represents the percentage of the F1-Score lost to sabotage that was successfully restored by the autoencoder.
    $$\text{Recovery Rate} = \frac{F1_{\text{defense}} - F1_{\text{attack}}}{F1_{\text{clean}} - F1_{\text{attack}}} \times 100$$

### Visualizing the Attack and Defenses
Below is a visual comparison of an original image, the FGSM perturbed image ($\epsilon = 0.1$), and the reconstructed outputs from our three defenses.

![Visual Comparison](path/to/your/saved_image_grid.png)

### Quantitative Performance
We evaluate the success of the defenses based off the target classifier on the reconstructed images, using the following metrics, based on the inner maximizer and outer minimizer framework analysis [\[5\]](#5):

### Assessing Adversarial Risk (Inner Maximizer)
| Risk Category           |   Sabotage (epsilon) |   Accuracy |   F1 (Macro) |
|-------------------------|----------------------|------------|--------------|
| Empirical Risk (Clean)  |                0.000 |      0.953 |        0.929 |
| Random Noise Baseline   |                0.020 |      0.950 |        0.926 |
| Adversarial Risk (FGSM) |                0.020 |      0.519 |        0.480 |

### Table 2: Comparative Analysis of Risk Mitigation (Outer Minimizer)
|   Sabotage (epsilon) |   Accuracy |   F1 (Macro) |   Precision |   Recall | Defense Strategy             | Risk Mitigation (%)   |
|----------------------|------------|--------------|-------------|----------|------------------------------|-----------------------|
|                0.020 |      0.519 |        0.480 |       0.507 |    0.482 | No Defense                   | 0.0%                  |
|                0.020 |      0.546 |        0.510 |       0.538 |    0.515 | Linear Blur                  | 6.7%                  |
|                0.020 |      0.640 |        0.600 |       0.620 |    0.604 | Standard AE                  | 26.8%                 |
|                0.020 |      0.845 |        0.805 |       0.823 |    0.803 | Adversarial U-Net (Proposed) | 72.5%                 |


![Training Curves](path/to/your/training_loss_graph.png)

### Model Confidence Shift
This graph visualizes how the classifier's confidence changes when subjected to the attack, and how confidence is restored by the U-Net defense.

> **PLACEHOLDER:** Insert Histogram/Boxplot showing classifier confidence scores across the different states.

![Confidence Shift Graph](path/to/your/confidence_graph.png)

---

## 🚀 Running the Pipeline

To execute the entire experiment:
1.  Run the **Imports and Setup** cells to install requirements (`tensorflow`, `opencv-python`, etc.).
2.  Run the **Data Loading** cells (the dataset will download automatically via `kagglehub`).
3.  Train or load the **Target Classifier**.
4.  Train the **Standard Autoencoder** and **U-Net Autoencoder** on the clean training set.
5.  Execute the **Evaluation Loop**, which generates FGSM samples on the test set, passes them through the three defenses, and records the metrics.

---

## 💡 Conclusion

*(Update this section based on your actual findings)*

* **Linear Blur** proved to be an inadequate defense, often dropping the baseline accuracy on clean images without significantly thwarting the FGSM attack.
* **The Standard Autoencoder** successfully removed some adversarial noise but suffered from detail loss (blurriness), which slightly degraded the classifier's performance.
* **The U-Net Autoencoder** provided the best trade-off. By utilizing skip connections, it successfully stripped the high-frequency FGSM perturbations while preserving the spatial integrity of the original images, resulting in the highest defense success rate.

## References

<a name="1"></a>
**[1]** Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). **Explaining and Harnessing Adversarial Examples.** *arXiv*. [https://doi.org/10.48550/arxiv.1412.6572](https://doi.org/10.48550/arxiv.1412.6572)

<a name="2"></a>
**[2]** Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). **The German Traffic Sign Recognition Benchmark: A multi-class classification competition.** *IJCNN*. [https://doi.org/10.1109/IJCNN.2011.6033395](https://doi.org/10.1109/IJCNN.2011.6033395)

<a name="3"></a>
**[3]** Liao, F., Liang, M., Dong, Y., Pang, T., Hu, X., & Zhu, J. (2018). **Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser.** *CVPR*.

<a name="4"></a>
**[4]** Meng, D., & Chen, M. (2017). **MagNet: a Two-Pronged Defense against Adversarial Examples.** *ACM CCS*. [https://doi.org/10.48550/arxiv.1705.09064](https://doi.org/10.48550/arxiv.1705.09064)

<a name="5"></a>
**[5]** Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). **Towards Deep Learning Models Resistant to Adversarial Attacks.** *ICLR*. [https://doi.org/10.48550/arxiv.1706.06083](https://doi.org/10.48550/arxiv.1706.06083)
