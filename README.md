# Norwegian & Danish: ADS vs. IDS — Vowel Space & Machine Learning Analyses 🎙️🇳🇴🇩🇰

Welcome to the **Norwegian & Danish ADS vs. IDS Analysis** repository! 🚀  
This project investigates how **Adult-Directed Speech (ADS)** and **Infant-Directed Speech (IDS)** differ acoustically and statistically across **Norwegian** and **Danish**, combining **phonetic analysis**, **mixed-effects modeling**, and **machine learning** approaches.

---

## 🎯 Project Overview

This repository includes:

- 🧩 **R scripts and notebooks** for statistical modeling (linear mixed-effects models, bootstrapped confidence intervals, etc.)  
- ⚙️ **Python utilities** for feature extraction, data preprocessing, and modeling  
- 📈 **Analysis workflows** for comparing vowel space area, variability, and model generalization  
- 📊 **Reproducible scripts** for generating figures and tables used in publication

---

## 🧠 Research Context

The project examines how **vowel space expansion** and **within-category variability** in IDS may enhance vowel learning and generalization.  
Analyses include:

- Comparison of vowel space area across **registers (ADS vs. IDS)**  
- Age-related effects and cross-linguistic comparisons (**Norwegian vs. Danish**)  
- Machine learning-based classification of vowels using **formant** and **MFCC** features  

These findings contribute to understanding how speech addressed to infants supports **phonetic category learning** and **word recognition**.

---

## ⚙️ Getting Started

### 1️⃣ Python Environment Setup

```bash
# Create and activate environment
conda create -n ads_ids python=3.10 -y
conda activate ads_ids

# Install dependencies
pip install numpy scipy pandas scikit-learn matplotlib librosa soundfile jupyter
---
```
### 2️⃣ R Environment Setup
```bash

Install the core R packages for statistical modeling:

install.packages(c(
  "lme4", "lmerTest", "emmeans", "broom.mixed",
  "car", "DHARMa", "ggplot2", "dplyr", "tidyr", "purrr",
  "readr", "stringr", "forcats", "rmarkdown"
))

Run the R Markdown analyses directly from RStudio or the terminal:

rmarkdown::render("Six_vowels_statistical_analysis_Norwegian_Danish_IDS_ADS_data.Rmd")

---
```
## 🧪 Typical Workflows

### 🔹 Vowel-Space Statistical Analysis (R)

1. Open `Six_vowels_statistical_analysis_...Rmd` or `Three_vowels_statistical_analysis_...Rmd`  
2. Knit or render the notebook  
3. View summary tables and plots (bootstrapped confidence intervals, regression trends)

### 🔹 Feature Extraction & Modeling (Python)

1. Set data paths in `paths.py`  
2. Extract features using the relevant feature extraction scripts  
3. Train or evaluate models with:

from model import train_model, evaluate  
from utils import load_dataset  

X_train, y_train = load_dataset(split="train")  
clf = train_model(X_train, y_train)  
evaluate(clf)

---

## 📁 Data Notes

- Raw audio and annotation data are **not included** in the repository.  
- Place your datasets under the directories defined in `paths.py`.  
- Ensure the file loading functions correctly join base paths with relative filenames.

---

## 🧩 Reproducibility

- Set all random seeds (`numpy`, `torch`, `random` in Python; `set.seed()` in R).  
- Track versions with `pip freeze` and `sessionInfo()`.  
- Save all results and plots under a dedicated `outputs/` directory.

---

## 📚 Citation

If you use or build upon this work, please cite:

@misc{ADS_IDS_Norw_Danish,  
  author       = {Arun Prakash Singh},  
  title        = {Norwegian and Danish: ADS vs. IDS Analyses},  
  year         = {2025},  
  howpublished = {\url{https://github.com/arunps12/NorwagianAndDanishADSandIDS}},  
  note         = {GPL-3.0 License}  
}

---

## 🤝 Contributing

Contributions are welcome!  
Please open an issue or pull request if you’d like to improve documentation, code clarity, or reproducibility.

## 🌟 About Me

Hi there! I'm **Arun Prakash Singh**, a **Marie Curie Research Fellow at the University of Oslo (UiO)**.  
My research focuses on **speech technology, data engineering, and machine learning**, with an emphasis on building intelligent, data-driven systems that model human communication and learning.  
I’m passionate about integrating **AI, analytics, and large-scale data pipelines** to advance our understanding of how humans process and acquire language.
