# CodingAtom_ML

SDN Traffic Classification using GAN Oversampling and 1D CNN

A leakage-safe machine learning pipeline for classifying Software-Defined Networking (SDN) traffic using a 1D Convolutional Neural Network (CNN) with GAN-based minority-class oversampling.

# Overview

The pipeline performs:

General data preprocessing

Removal of string/object features

Target leakage detection

Train/test splitting

Training-only imputation and standardization

GAN-based oversampling of the minority class

1D CNN training

Evaluation using a classification report and confusion matrix

Workflow

sdn.csv
   |
   v
Preprocessing
   |
   v
Remove String Features
   |
   v
Leakage Check
   |
   v
Train / Test Split
   |
   +--------------------+
   |                    |
   v                    v
Training Data        Test Data
   |                    |
   v                    |
Imputation + Scaling   |
   |                    |
   v                    |
GAN Oversampling       |
   |                    |
   v                    |
1D CNN Training        |
   |                    |
   +---------+----------+
             |
             v
      Classification Report

Project Structure

SDN-GAN-CNN/
|
├── sdn.csv
├── gan_cnn_sdn_pipeline.py
├── cnn_gan_classification_report.txt
└── README.md

If the dataset is too large for GitHub, use Git LFS or provide a download link instead of committing the CSV directly.

# Requirements

Python 3.9+

NumPy

Pandas

Scikit-learn

PyTorch

Install the dependencies:

pip install numpy pandas scikit-learn torch

For practical training speed, a CUDA-enabled PyTorch installation and NVIDIA GPU are recommended.

Running the Project

Place sdn.csv in the same directory as the Python script and run:

python gan_cnn_sdn_pipeline.py

The script prints the detected device, removed features, class distributions, training progress, classification report, and confusion matrix.

# 1. Data Preprocessing

The dataset is loaded with Pandas:

df = pd.read_csv(DATA)

Infinite values are converted to missing values and duplicate rows are removed:

df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()

Rows without a target label are removed.

The target column is:

label

# 2. Removing String Features

The original dataset contains non-numerical fields such as:

src
dst
Protocol

These are removed because the CNN receives numerical tensors.

string_cols = df.drop(columns=[TARGET]).select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

df = df.drop(columns=string_cols)

The remaining features are converted to numeric values where possible.

# 3. Leakage Check

The earlier XGBoost experiment produced approximately 100% accuracy, which is unusually high for network-traffic classification.

The new pipeline therefore checks for obvious target leakage.

It checks whether a feature:

Is an exact copy of the target

Perfectly separates the classes

Features satisfying these definite leakage conditions are removed.

The pipeline deliberately does not remove every feature that is merely correlated with the target, because legitimate network measurements can naturally be highly predictive.

For example:

pktcount
bytecount
pktrate
byteperflow

may legitimately contain information about network traffic.

# 4. Train/Test Split

The split occurs before GAN oversampling and before fitting preprocessing:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

The dataset is divided into:

80% Training
20% Testing

The test set is kept completely untouched during GAN training.

This is one of the most important parts of the leakage-safe design.

# 5. Imputation and Scaling

Missing values are replaced using the training-set median:

imputer = SimpleImputer(strategy="median")

Features are then standardized:

scaler = StandardScaler()

Both transformations are fitted only on training data:

X_train = scaler.fit_transform(
    imputer.fit_transform(X_train)
)

The test data only uses the already-fitted transformations:

X_test = scaler.transform(
    imputer.transform(X_test)
)

This prevents information from the test set leaking into training.

# 6. GAN Oversampling

If the dataset is imbalanced, the minority class may be harder for the classifier to learn.

The minority class is identified from the training set:

classes, counts = np.unique(y_train, return_counts=True)
minority = classes[np.argmin(counts)]

Only minority-class training samples are used to train the GAN.

Generator

The Generator takes random latent noise and produces synthetic feature vectors:

Random Noise
     |
     v
Linear
     |
   ReLU
     |
     v
Linear
     |
   ReLU
     |
     v
Synthetic Network Features

Discriminator

The Discriminator tries to distinguish real minority samples from generated samples:

Network Features
      |
      v
   Linear
      |
 LeakyReLU
      |
      v
   Linear
      |
      v
 Sigmoid
      |
      v
 Real / Fake

The two networks are trained adversarially.

After training, the Generator produces additional minority-class samples.

# 7. Augmented Training Data

The generated samples are combined with the original training data:

X_aug = np.vstack([X_train, X_syn])
y_aug = np.concatenate([y_train, y_syn])

The training data is shuffled afterward.

The test set is never augmented.

Therefore:

GAN → Training data only
CNN → Augmented training data
Evaluation → Original untouched test data

# 8. 1D CNN Classifier

The classifier is a one-dimensional convolutional neural network.

The input is reshaped to:

(samples, channels, features)

with one input channel.

The CNN uses several convolutional layers:

nn.Conv1d(1, 32, 3, padding=1)
nn.Conv1d(32, 64, 3, padding=1)
nn.Conv1d(64, 96, 3, padding=1)

These layers learn patterns from the numerical feature representation.

Adaptive average pooling reduces the representation before the fully connected classification layers.

The final layer outputs the predicted class.

# 9. CNN Training

The optimizer is AdamW:

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

The model uses cross-entropy loss:

criterion = nn.CrossEntropyLoss()

The current configuration trains for:

10 and 100 epochs
Batch size: 1024
Learning rate: 0.001

These values can be adjusted depending on available GPU memory and model performance.

# 10. GPU Support

The script automatically detects CUDA:

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

Therefore:

CUDA available → NVIDIA GPU
CUDA unavailable → CPU

Because GAN training and CNN training can be computationally expensive for a large dataset, GPU execution is recommended.

# 11. Evaluation

The final model is evaluated only on the untouched test set.

The script generates:

Precision

Recall

F1-score

Accuracy

Confusion matrix

Classification report:

classification_report(
    y_test,
    pred,
    digits=4
)

Confusion matrix:

confusion_matrix(
    y_test,
    pred
)

The confusion matrix has the form:

                 Predicted
                 0       1

Actual 0        TN      FP
Actual 1        FN      TP

# Why Leakage Prevention Matters

A common mistake in imbalanced classification is to oversample before splitting the dataset.

Incorrect:

Dataset
   |
GAN / SMOTE
   |
Train/Test Split

This can allow information from synthetic training examples to overlap with the eventual test set.

The correct approach used here is:

Dataset
   |
Train/Test Split
   |
   +------------------+
   |                  |
Training             Test
   |
Preprocessing
   |
GAN
   |
CNN
   |
Evaluation on Test

This gives a much more reliable estimate of generalization.

Dataset Features

The dataset contains SDN/network-flow measurements including:

dt
switch
src
dst
pktcount
bytecount
dur
dur_nsec
tot_dur
flows
packetins
pktperflow
byteperflow
pktrate
Pairflow
Protocol
port_no
tx_bytes
rx_bytes
tx_kbps
rx_kbps
tot_kbps
label

The string features:

src
dst
Protocol

are removed during preprocessing.

The target is:

label

Results

The exact classification metrics should be generated by running the supplied Python script.

# Example output format:

CLASSIFICATION REPORT

              precision    recall  f1-score   support

           0     0.xxxx    0.xxxx    0.xxxx      xxxx
           1     0.xxxx    0.xxxx    0.xxxx      xxxx

    accuracy                         0.xxxx     xxxxx
   macro avg     0.xxxx    0.xxxx    0.xxxx     xxxxx
weighted avg     0.xxxx    0.xxxx    0.xxxx     xxxxx

The exact values can vary between environments and training runs.

The earlier 100% XGBoost result should not be treated as the final result until the leakage investigation and leakage-safe pipeline have been evaluated.

Possible Future Improvements

Hyperparameter optimization

Early stopping

Learning-rate scheduling

ROC-AUC

PR-AUC

ROC and Precision-Recall curves

GAN sample-quality evaluation

SHAP explainability

Feature importance analysis

Comparison with Random Forest

Comparison with XGBoost

Comparison with MLP

Comparison between GAN and SMOTE

Cross-validation

Ablation studies

A useful experiment would be:

XGBoost
   vs
CNN
   vs
CNN + SMOTE
   vs
CNN + GAN

This would make the project considerably stronger as an ML research/academic project.

# Disclaimer

This project is intended for educational, research, and machine-learning experimentation purposes. Model performance should be validated on independent data before deployment in a real-world SDN security environment.
