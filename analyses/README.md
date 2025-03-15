# Whoop Recovery Score Prediction

This directory contains analysis scripts, models, and visualizations for predicting Whoop recovery scores using Garmin data.

## Whoop Recovery Predictor

The `predict_whoop_recovery.py` script contains a model to predict Whoop recovery scores using primary variables from Garmin data. The model uses the following features:

1. **bodyBatteryMostRecentValue** - Garmin's equivalent to Whoop's recovery score
2. **restingHeartRate** - Morning resting heart rate
3. **sleepingSeconds** - Total sleep duration
4. **sleep_score** - Garmin's assessment of sleep quality
5. **averageStressLevel** - Garmin's stress measurement (derived from HRV)

## Requirements

The script requires:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

These should be included in the project's main requirements.txt file.

## Usage

### Running the Script

To train the model and generate visualizations, run:

```bash
cd analyses
python predict_whoop_recovery.py
```

This will:
1. Train a Random Forest model on your data
2. Save the trained model to `whoop_recovery_model.joblib` in the analyses directory
3. Generate and save visualizations to the `analyses/visualizations/` folder
4. Print model performance metrics
5. Show an example prediction using sample data

### Using the Predictor Class in Other Scripts

```python
from analyses.predict_whoop_recovery import WhoopRecoveryPredictor

# Create a new predictor and train it
predictor = WhoopRecoveryPredictor()
metrics = predictor.train()

# Or load a previously trained model
predictor = WhoopRecoveryPredictor(model_path='analyses/whoop_recovery_model.joblib')

# Make a prediction with new Garmin data
garmin_data = {
    'bodyBatteryMostRecentValue': 65,
    'restingHeartRate': 52,
    'sleepingSeconds': 27000,  # 7.5 hours
    'sleep_score': 80,
    'averageStressLevel': 25
}
predicted_recovery = predictor.predict(garmin_data)
print(f"Predicted Whoop recovery score: {predicted_recovery:.2f}")
```

## Model Performance

The model performance metrics will vary based on your personal data. After training, the script will display:
- R² score (coefficient of determination) for both training and test sets
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Feature importance rankings

## Visualizations

The script generates two visualizations saved to the `analyses/visualizations/` folder:
1. `feature_importance.png` - Shows which Garmin variables are most predictive of Whoop recovery scores
2. `actual_vs_predicted.png` - Scatterplot showing how well the model's predictions match actual Whoop recovery scores 