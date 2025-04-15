ADSC3910 Project

Classification of News Category

Welcome to our project, Classification of News Category. This project aims to classify global news articles using the News Category Dataset. The primary objective is to automatically identify the category of each article, focusing on key areas such as Politics, Sports, Health, Business, and Entertainment. Accurate classification enhances information retrieval and improves user experience in news applications.

🔍 Project Sections

- [Team](#team)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)

👥 Team

Our team comprises of Post Baccalaureate Data Science students at the Thomposon River University in collaboration with [Quan Nguyen](https://github.com/quan3010)

- [Khadiza Tannee](https://github.com/Tannee-Siddique)– Prepared and preprocessed the news data for model training and evaluation
- [Viswateja Adothi](https://github.com/viswatejaadothi)- Developed and fine-tuned machine learning models for news classification.
- [Solomon Maccarthy](https://github.com/FiiMac)- Handled training, performance evaluation, and model validation.

📚 Project Overview

Classifying the various news categories is the goal of this study. We want to employ models to assist in  knowing the various news categories and enhances users experience on news platforms, improves content recommendations, and aids in content moderation

⚙️ Installation

Ensure you have the following tools installed
- [VS Code](https://code.visualstudio.com/)

💻 Platform
- [Databricks](https://www.databricks.com/)
- [Mongodb](https://account.mongodb.com/)

#### Guest Credentials

- {
    "host": "clusterx.xxxxx.mongodb.net",
    "port": "27017",
    "username": "Guest1234",
    "password": "Group_1"
  }

🚀 Usage

- [README.md](https://github.com/TRU-PBADS/adsc3910-project-group-1/blob/main/README.md)

### Local Setup

Follow the instructions below to run the prediction pipeline locally.

1. Clone the repo:

```bash
git clone https://github.com/TRU-PBADS/adsc3910-project-group-1.git
```

2.  Activate the environment:

```bash
conda env create --file adsc_3910_group_1_env.yaml
conda activate adsc_3910_group_1
```

3. Run the model: Since the data has already been preprocessed and credentials are provided, run:

```bash
python script/cross_validate.py
python script/evaluate.py
python script/pre_processing.py
python script/predict.py
python script/train.py
```

4. Raw data required to run the pipeline is already downloaded and saved to [folder](https://github.com/TRU-PBADS/adsc3910-project-group-1/tree/main/datasets)


5. Upload raw data to Mongodb using [export_to_mongodb.ipynb](https://github.com/TRU-PBADS/adsc3910-project-group-1/blob/main/datasets/news_category_dataset_v3.json)

6. Now you can run the pipeline [pipeline.ipynb](https://github.com/TRU-PBADS/adsc3910-project-group-1/blob/main/Notebooks/data_preprocessing/mongodb_pipeline.ipynb)

7. Run to test some scripts using the file in [test_preprocessing.py](https://github.com/TRU-PBADS/adsc3910-project-group-1/tree/main/script/test_preprocessing.py)

- `N_ESTIMATORS`: This parameter denotes the number of learning rate (2e-5) with the AdamW optimizer. The model was trained for 3 epochs. Batch size of 16 was used. Model uses a cross-entropy loss which technically is the default loss function for classification. A 5 fold cross-validation is used to access the model's performance across multiple data splits. Each part being used as a validation set once, while the other four parts were used for training.



📦 Dependencies

All required Python packages and environment setup are listed in the [adsc_3910_group_1_env.yaml](https://github.com/TRU-PBADS/adsc3910-project-group-1/adsc_3910_group_1_env.yaml) file.
