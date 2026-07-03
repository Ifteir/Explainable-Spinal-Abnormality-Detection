# SpinalAbnormalityDetection# Explainable Machine Learning for Reliable Spinal Abnormality Detection

## Overview

This project is a Data Mining and Machine Learning-based study focused on the detection of spinal abnormalities using biomechanical spine data. The project applies supervised machine learning techniques to classify spinal conditions and uses explainable artificial intelligence methods to improve model transparency and interpretability.

The main objective of this project is not only to build an accurate classification model but also to explain how important biomechanical features contribute to the prediction of spinal abnormality. This makes the system more reliable and useful for healthcare-related decision support.

## Project Title

**Explainable Machine Learning for Reliable Spinal Abnormality Detection**

## Subject

**Data Mining**

## Project Objectives

* To analyze biomechanical spine data for abnormality detection.
* To apply machine learning classification algorithms.
* To evaluate model performance using standard evaluation metrics.
* To use explainable AI techniques for model interpretation.
* To identify the most influential features affecting spinal abnormality prediction.
* To present results using visualizations such as heatmaps, boxplots, SHAP plots, confusion matrix, and calibration analysis.

## Dataset

The project uses a spine dataset containing biomechanical features related to spinal structure and condition. The dataset includes measurements that help distinguish between normal and abnormal spinal cases.

### Dataset File

```text
Dataset_spine.csv
```

## Features Used

The dataset contains biomechanical attributes such as:

* Pelvic incidence
* Pelvic tilt
* Lumbar lordosis angle
* Sacral slope
* Pelvic radius
* Degree of spondylolisthesis
* Class label for spinal condition

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP
* Machine Learning Classification Models

## Project Structure

```text
Explainable-Spinal-Abnormality-Detection/
│
├── Code/
│   ├── spine.py
│   ├── Dataset_spine.csv
│   ├── model_performance.csv
│   ├── Figure1_Correlation_Heatmap.png
│   ├── Figure2_Boxplot_Degree_Spondylolisthesis.png
│   ├── Figure2_Boxplot_Lumbar_Lordosis_Angle.png
│   ├── Figure2_Boxplot_Pelvic_Incidence.png
│   ├── Figure3_SHAP_Summary.png
│   ├── Supp_Calibration.png
│   ├── Supp_Confusion_Matrix.png
│   ├── README.md
│   └── LICENSE
│
├── Dataset_spine.csv
└── Explainable Machine Learning for Reliable Spinal Abnormality Detection.pdf
```

## Methodology

The project follows a complete data mining workflow:

1. **Data Collection**
   The spine dataset is loaded and prepared for analysis.

2. **Data Preprocessing**
   Missing values, feature types, and class labels are checked before model training.

3. **Exploratory Data Analysis**
   Correlation analysis and boxplots are used to understand the relationship between biomechanical features and spinal abnormality classes.

4. **Model Training**
   Machine learning classification models are trained on the dataset.

5. **Model Evaluation**
   The models are evaluated using classification metrics such as accuracy, precision, recall, F1-score, confusion matrix, and calibration analysis.

6. **Explainability Analysis**
   SHAP analysis is used to explain the contribution of each feature in the prediction process.

## Results

The project generates several output files and visualizations, including:

* Correlation heatmap
* Feature-based boxplots
* SHAP summary plot
* Confusion matrix
* Calibration curve
* Model performance comparison table

These results help evaluate both the predictive performance and interpretability of the machine learning model.

## Important Visualizations

### Correlation Heatmap

```text
Figure1_Correlation_Heatmap.png
```

Shows the relationship between different biomechanical features.

### Boxplots

```text
Figure2_Boxplot_Degree_Spondylolisthesis.png
Figure2_Boxplot_Lumbar_Lordosis_Angle.png
Figure2_Boxplot_Pelvic_Incidence.png
```

These plots compare important feature distributions across different spinal condition classes.

### SHAP Summary Plot

```text
Figure3_SHAP_Summary.png
```

Shows the impact of each feature on the model prediction and improves model interpretability.

### Confusion Matrix

```text
Supp_Confusion_Matrix.png
```

Displays the classification performance of the trained model.

### Calibration Plot

```text
Supp_Calibration.png
```

Shows how well the predicted probabilities match the actual outcomes.

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Explainable-Spinal-Abnormality-Detection.git
```

### Step 2: Open the Project Folder

```bash
cd Explainable-Spinal-Abnormality-Detection
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
```

### Step 4: Run the Python File

```bash
cd Code
python spine.py
```

## Project Paper

The complete project paper is included in this repository:

```text
Explainable Machine Learning for Reliable Spinal Abnormality Detection.pdf
```

The paper explains the background, methodology, results, visualizations, and conclusion of the project.

## Applications

This project can be useful for:

* Data mining coursework
* Machine learning classification practice
* Healthcare analytics
* Explainable AI research
* Medical decision-support system development
* Feature importance analysis using SHAP

## Conclusion

This project demonstrates how data mining and machine learning techniques can be used for reliable spinal abnormality detection. By combining classification models with explainable AI methods, the project provides both predictive performance and interpretability. The use of SHAP analysis makes the model more transparent by identifying the most important biomechanical features involved in the classification process.

## Author

**Ifteir Hossain**
Department of Computer Science and Engineering
United International University
Batch: 223

## License

This project is created for academic and educational purposes.

## Acknowledgement

This project was completed as part of the Data Mining course under the Department of Computer Science and Engineering at United International University.
