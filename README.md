# Sri Lankan Dengue Prediction — NGBoost deployment

Experimental historical forecast for January 2021, not a current public-health forecasting service.

## Required Colab exports
Download these from Section 10 of `Sri_Lanka_Dengue_NGBoost_Fixed_Complete_Colab.ipynb` and place them beside `app.py`:

- `dengue_ngboost_bundle.joblib` (required; trained NGBoost model, preprocessor and ordered feature list)
- `dengue_ngboost_metadata.json` (required)
- `dengue_ngboost_historical_predictions.csv` (recommended for historical evaluation graph)

From Section 8 also upload:

- `dengue_ngboost_permutation_importance.csv` (recommended for feature importance graph)

The dataset is at `data/dengue_history.xlsx` (check licensing before making the repository public).

## Deploy to existing Streamlit Community Cloud app

1. Open the existing GitHub repository `thilankaathuraliya/sri-lanka-dengue-forecast-app`.
2. Upload the new `app.py`, `requirements.txt`, `README.md`, and the exported NGBoost files to the **repository root**. Replace existing files when prompted.
3. Keep the `data/dengue_history.xlsx` file in `data/`.
4. Keep branch `main` and main file path `app.py` in Streamlit Community Cloud.
5. Streamlit should redeploy after the GitHub commit. Use **Manage app → Reboot app** if necessary.
6. Test Colombo January 2021. The Colab exported forecast had median prediction about 127.51, lower about 64.74, upper about 250.20, baseline 92. Small differences should be investigated.

**Important:** The `.joblib` file was created in Colab and is not included here. Never load untrusted joblib files. If unpickling fails due to incompatible Python/scikit-learn/NGBoost versions, pin dependencies to the versions used in Colab (record them using `pip show ngboost scikit-learn numpy scipy`).

## Notes

- NGBoost model was fitted on data through 2018-12; historical 2019–2020 evaluation is not blind.
- The 90% interval had only 69.2% empirical coverage on that historical evaluation.
- Do not present these forecasts as validated public-health guidance.
