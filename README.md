# adsc3910_project


# Classification of News Category
Welcome to our project, the Classification of News Category, This aims to explore and classify the type of new categories globally by leveraging the News Category Dataset. The primary objective is to automatically identify the classification of each article, focusing on key categories such as Politics, Sports, Health, Business and Entertainment. Analyzing these categories with models can accurately categorize news articles, enhancing information retrieval and user experience in news applications.

Dive into the sections below to discover more about our project:

- [Team](#team)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)

## Team

Our team comprises of Post Baccalaureate Data Science students at the Thomposon River University in collaboration with [Quan Nguyen](https://github.com/quan3010)

- [Khadiza Tannee](https://github.com/Tannee-Siddique)
- [Viswateja Adothi](https://github.com/viswatejaadothi)
- [Solomon Maccarthy](https://github.com/FiiMac)

## Project Overview

Classifying the various news categories is the goal of this study. We want to employ models to assist in  knowing the various news categories and enhances users experience on news platforms, improves content recommendations, and aids in content moderation
## Installation

Ensure you have the following tools installed

- [Mongodb](https://account.mongodb.com/)
- [VS Code](https://code.visualstudio.com/)
- [Databricks](https://www.databricks.com/)


## Usage

- [README.md](https://github.com/TRU-PBADS/adsc3910-project-group-1/blob/main/README.md)

### Local Setup

Follow the instructions below to run the prediction pipeline locally.

1. Clone the repo:

```bash
git clone https://github.com/TRU-PBADS/adsc3910-project-group-1.git
```

2.  and activate the required environment:

```bash
conda env create --file adsc_3910_group_1_env.yaml
conda activate adsc_3910_group_1
```

3. Reset and clean the existing analysis results from directories by running the below command from the project root directory:

```bash
make clean
```
4. Raw data required to run the pipeline is already downloaded and saved to [folder](https://github.com/TRU-PBADS/adsc3910-project-group-1/tree/main.)

5. Run the model using the files in [folder](https://github.com/TRU-PBADS/adsc3910-project-group-1/tree/main.)

- `N_ESTIMATORS`: This parameter denotes the number of learning rate (2e-5) with the AdamW optimizer. The model was trained for 3 epochs. Batch size of 16 was used. Model uses a cross-entropy loss which technically is the default loss function for classification. A 5 fold cross-validation is used to access the model's performance across multiple data splits. Each part being used as a validation set once, while the other four parts were used for training.


### Dependencies

For the Python dependencies and the conda environment creation file, please check [here](https://github.com/TRU-PBADS/adsc3910-project-group-1/adsc_3910_group_1_env.yaml)