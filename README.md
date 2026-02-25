# Adversarial Robustness Study on CNN Models using FGSM

## 1. Project Overview

This project presents a systematic experimental study on the robustness of Convolutional Neural Networks (CNNs) against adversarial attacks. The study focuses on the Fast Gradient Sign Method (FGSM) to evaluate how small, human-imperceptible perturbations can significantly degrade model performance.

The objective is to analyze model vulnerability and improve robustness using adversarial training techniques.

---

## 2. Problem Statement

Deep learning models, particularly CNNs, achieve high accuracy on image classification tasks. However, they are vulnerable to adversarial perturbations—carefully crafted noise that can cause incorrect predictions while remaining visually indistinguishable to humans.

This project investigates:

- How vulnerable is a standard CNN to FGSM attacks?
- How does adversarial training improve robustness?
- What is the trade-off between clean accuracy and adversarial accuracy?

---

## 3. Dataset

- Dataset Used: MNIST Handwritten Digits
- Number of Classes: 10
- Image Size: 28x28 grayscale
- Training Samples: 60,000
- Test Samples: 10,000

---

## 4. Methodology

### 4.1 Baseline Model

- Convolutional Neural Network (CNN)
- ReLU activations
- MaxPooling layers
- Fully connected classifier
- Cross-entropy loss
- Optimizer: Adam

### 4.2 Adversarial Attack

Fast Gradient Sign Method (FGSM)

FGSM generates adversarial examples using:

    x_adv = x + epsilon * sign(gradient(loss, x))

Where:
- epsilon controls perturbation strength
- gradient is computed with respect to input

### 4.3 Adversarial Training

The model is retrained using a mixture of:
- Clean samples
- FGSM-generated adversarial samples

This improves robustness against adversarial attacks.

---

## 5. Experimental Results

| Model Type        | Clean Accuracy | Adversarial Accuracy |
|-------------------|---------------|----------------------|
| Baseline CNN      | ~98%          | ~40%                 |
| Adversarially Trained CNN | ~97–98% | ~85–90%             |

### Observations

- The baseline model performs well on clean data but fails under FGSM attack.
- Adversarial training significantly improves robustness.
- There is a slight trade-off between clean accuracy and adversarial accuracy.

---

## 6. Key Learnings

- Deep neural networks are highly vulnerable to gradient-based adversarial attacks.
- Small perturbations can drastically change predictions.
- Adversarial training is an effective defense mechanism.
- Robustness and accuracy must be balanced in safety-critical applications.

---

## 7. Technologies Used

- Python
- PyTorch
- Google Colab
- NumPy
- Matplotlib

---

## 8. Future Work

- Evaluate robustness under stronger attacks (PGD, BIM)
- Test transferability of adversarial examples
- Apply study on CIFAR-10 dataset
- Explore certified robustness techniques

---

## 9. Conclusion

This project demonstrates the inherent vulnerability of CNN models to adversarial attacks and validates adversarial training as an effective defense strategy. The findings highlight the importance of robustness evaluation in modern deep learning systems, especially in security-sensitive domains.

---

## Author

Final Year B.Tech CSE (AI & ML) Student  
Adversarial Machine Learning Research Project
