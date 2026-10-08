# Antimicrobial Resistance Trends in Europe (Acinetobacter / Fluoroquinolones)

An exploratory machine learning project modeling how often *Acinetobacter* bacteria are resistant to fluoroquinolone antibiotics across European countries.

## Question
Can year and country information be used to model the percentage of *Acinetobacter* isolates resistant to fluoroquinolones in European countries?

## Why it matters
*Acinetobacter* is a bacteria that causes hard-to-treat hospital infections, and resistance to common antibiotics like fluoroquinolones makes those infections more dangerous. Resistance rates vary a lot between countries, which is part of what this project explores.

## Data
- **Source:** ECDC (European Centre for Disease Prevention and Control) Surveillance Atlas of Infectious Diseases
- **Indicator:** percentage of *Acinetobacter spp.* isolates resistant to fluoroquinolones
- **Coverage:** EU/EEA countries, 2012–2023 (358 rows before cleaning)
- **Cleaning:**
  - Dropped columns that were the same for every row (topic, unit, indicator)
  - Removed missing values (marked as "-" in the data)
  - Converted percentages to a 0–1 scale
  - Recoded years as numbers starting from 0 (2011 = 0)
- After cleaning, 26 countries were left. The United Kingdom was dropped because "United Kingdom" isn't in the Word2Vec vocabulary as typed (the model uses "United_Kingdom").

## Method
**Representing countries:** I used pretrained Word2Vec embeddings (Google News, 300 dimensions) to turn each country name into a vector of numbers. The idea was that countries with similar embeddings might have similar resistance patterns.

**Models I tried:**
1. **Linear regression** using year and the first value of each country's embedding
2. **Support vector regression (SVR)** with the same two features
3. **Decision tree**, first with max depth 10, then tuned with GridSearchCV (5-fold cross-validation)
4. **Keras neural network** using year plus the full 300-number embedding, with dropout, L2 regularization, and early stopping

**Tools:** Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, gensim (Word2Vec), matplotlib, seaborn

## Results

| Model | How it was evaluated | MSE | R² | RMSE (percentage points) |
|---|---|---|---|---|
| Linear regression | Training data (in-sample) | 0.109 | 0.19 | ~33 |
| SVR | Training data (in-sample) | 0.122 | 0.09 | ~35 |
| Decision tree (max depth 10) | Random 80/20 split | 0.0044 | 0.97 | ~6.6 |
| Decision tree (GridSearchCV) | Random 80/20 split | 0.0047 | 0.97 | ~6.8 |
| Neural network | Random 80/20 split | [fill in] | – | – |

Best GridSearchCV settings: max_depth = 11, min_samples_leaf = 1, min_samples_split = 2

**What I learned from the results:**
- Linear regression and SVR did poorly. Resistance levels depend much more on the country than on the year, and a straight line can't capture that.
- The decision trees scored much higher, but mostly because they learned each country's typical resistance level. The test set was a random split, so it included years in between years the model had already seen for the same country. This means the high R² reflects fitting historical data, not predicting the future.
- I also used the neural network to generate predictions for each country out to 2086 (saved in `predictions.csv` and `country_predictions/`). These were an experiment in building a prediction pipeline and shouldn't be taken seriously. Twelve years of data can't support forecasts 60+ years out, and some predictions can go above 100%.

## Limitations
- **Small dataset:** only around 12 years per country, with some years missing.
- **Word2Vec embeddings reflect how country names appear in news articles**, not anything about health systems or antibiotic use, so they're not a meaningful health feature.
- **The linear regression, SVR, and decision tree models only used the first of the 300 embedding values**, which basically works as an arbitrary ID number for each country.
- **Linear regression and SVR were evaluated on their own training data**, so their scores are optimistic.
- **The random train/test split mixes years**, so none of the scores measure true forecasting ability.
- **There's no genomic data in this project**, even though the notebook file name says "genome." It's country-level surveillance percentages.

## Next steps
- Use a **time-based split** (train on 2012–2020, test on 2021–2023) to measure real forecasting performance
- Replace Word2Vec with **actual country features**, like antibiotic consumption (also available from ECDC) or healthcare spending
- Use **one-hot encoding** for countries instead of a single embedding value
- Keep predictions within a few years of the data and clip them between 0% and 100%
- Fix the UK by using the "United_Kingdom" token

## Files
- `AMRGENOMEpred.ipynb`: the full notebook (cleaning, models, and predictions)
- `AMR.csv`: the ECDC data used (source: ECDC Surveillance Atlas of Infectious Diseases)
- `predictions.csv` and `country_predictions/`: experimental long-range predictions from the neural network

## How to run it
1. Open `AMRGENOMEpred.ipynb` in Google Colab.
2. Upload `AMR.csv`.
3. Run all cells. The Word2Vec model is about 1.6 GB, so it takes a few minutes to download the first time.
