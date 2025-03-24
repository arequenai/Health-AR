import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import joblib
import sys
sys.path.append('..')  # Add parent directory to path

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths
ANALYSES_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZATIONS_DIR = os.path.join(ANALYSES_DIR, 'visualizations')
MODELS_DIR = os.path.join(ANALYSES_DIR, 'models')

# Ensure directories exist
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Import config from ETL package if available
try:
    from etl import config
    # If imported from config, make sure paths are absolute
    GARMIN_DAILY_FILE = os.path.join(PROJECT_ROOT, 'data', 'garmin_daily.csv')
    WHOOP_FILE = os.path.join(PROJECT_ROOT, 'data', 'whoop.csv')
    DAILY_METRICS_FILE = os.path.join(ANALYSES_DIR, 'daily_metrics.csv')
except ImportError:
    # Default paths if config module is not available
    GARMIN_DAILY_FILE = os.path.join(PROJECT_ROOT, 'data', 'garmin_daily.csv')
    WHOOP_FILE = os.path.join(PROJECT_ROOT, 'data', 'whoop.csv')
    DAILY_METRICS_FILE = os.path.join(ANALYSES_DIR, 'daily_metrics.csv')

# Default model file path
MODEL_FILE = os.path.join(MODELS_DIR, 'whoop_recovery_predictor.joblib')

print(f"Using data files: \n- Garmin: {GARMIN_DAILY_FILE}\n- Whoop: {WHOOP_FILE}\n- Daily Metrics: {DAILY_METRICS_FILE}")
print(f"Visualizations will be saved to: {VISUALIZATIONS_DIR}")
print(f"Models will be saved to: {MODELS_DIR}")

class WhoopRecoveryPredictor:
    """
    A class to predict Whoop recovery scores using Garmin data.
    """
    
    def __init__(self, model_type='Ridge', verbose=True):
        self.verbose = verbose
        self.model_type = model_type
        
        # Create the Ridge model pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
        
        # These features are identified as the most important
        self.feature_names = [
            'restingHeartRate', 
            'sleep_score', 
            'avg_sleep_stress',
            'max_sleep_body_battery'
        ]
        
        self.feature_bounds = {
            'restingHeartRate': (30, 120),
            'sleep_score': (0, 100),
            'avg_sleep_stress': (0, 100),
            'max_sleep_body_battery': (0, 100)
        }
    
    def load_data(self, verbose=True):
        """
        Load and preprocess data from Garmin, Whoop, and daily metrics.
        
        Args:
            verbose (bool): Whether to print data quality information
            
        Returns:
            tuple: (X, y) features and target for training
        """
        # Load data
        df_garmin = pd.read_csv(GARMIN_DAILY_FILE)
        df_whoop = pd.read_csv(WHOOP_FILE)
        df_daily_metrics = pd.read_csv(DAILY_METRICS_FILE)
        
        if verbose:
            print(f"\nData Preprocessing Report:")
            print(f"- Garmin records: {len(df_garmin)}")
            print(f"- Whoop records: {len(df_whoop)}")
            print(f"- Daily metrics records: {len(df_daily_metrics)}")
        
        # Convert date columns to datetime for proper merging
        df_garmin['date'] = pd.to_datetime(df_garmin['date'])
        df_whoop['date'] = pd.to_datetime(df_whoop['date'])
        df_daily_metrics['date'] = pd.to_datetime(df_daily_metrics['date'])
        
        # Check for missing dates in each dataset
        garmin_dates = set(df_garmin['date'].dt.strftime('%Y-%m-%d'))
        whoop_dates = set(df_whoop['date'].dt.strftime('%Y-%m-%d'))
        metrics_dates = set(df_daily_metrics['date'].dt.strftime('%Y-%m-%d'))
        
        # Find dates in one dataset but not the other
        dates_only_in_garmin = garmin_dates - whoop_dates
        dates_only_in_whoop = whoop_dates - garmin_dates
        dates_only_in_metrics = metrics_dates - whoop_dates
        
        if verbose and (dates_only_in_garmin or dates_only_in_whoop or dates_only_in_metrics):
            print(f"- Warning: {len(dates_only_in_garmin)} dates in Garmin but not in Whoop")
            print(f"- Warning: {len(dates_only_in_whoop)} dates in Whoop but not in Garmin")
            print(f"- Warning: {len(dates_only_in_metrics)} dates in metrics but not in Whoop")
        
        # Merge datasets on date (inner join keeps only dates present in all datasets)
        merged_df = pd.merge(df_garmin, df_whoop[['date', 'recovery_score']], on='date', how='inner')
        
        # Get all available columns from daily_metrics (except date which we're merging on)
        daily_metrics_columns = [col for col in df_daily_metrics.columns if col != 'date']
        
        # Merge with all available metrics
        merged_df = pd.merge(merged_df, df_daily_metrics[['date'] + daily_metrics_columns], 
                           on='date', how='inner')
        
        if verbose:
            print(f"- After merge: {len(merged_df)} records")
        
        # Filter out days with sleep_score = 0 (invalid/missing data)
        zero_sleep_count = len(merged_df[merged_df['sleep_score'] == 0])
        if zero_sleep_count > 0:
            if verbose:
                print(f"- Filtering out {zero_sleep_count} days with sleep_score = 0")
            merged_df = merged_df[merged_df['sleep_score'] > 0]
        
        # Filter feature names to only include columns that exist in the merged dataset
        available_features = [f for f in self.feature_names if f in merged_df.columns]
        
        if verbose:
            if len(available_features) < len(self.feature_names):
                missing_features = set(self.feature_names) - set(available_features)
                print(f"\nSome features are not available in the dataset:")
                for feature in missing_features:
                    print(f"- {feature} is not available")
                print(f"Using {len(available_features)} out of {len(self.feature_names)} possible features")
        
        # Update feature_names to only use available features
        self.feature_names = available_features
        
        # Check for missing values before cleaning
        missing_values = merged_df[self.feature_names + ['recovery_score']].isnull().sum()
        if verbose and missing_values.sum() > 0:
            print("\nMissing values before cleaning:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"- {col}: {count} missing values")
        
        # Drop rows with any missing values in our features or target
        merged_df = merged_df.dropna(subset=self.feature_names + ['recovery_score'])
        
        if verbose:
            print(f"- After dropping missing values: {len(merged_df)} records")
        
        # Check for outliers using IQR method
        def identify_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            return outliers
        
        # Report outliers for each feature
        if verbose:
            print("\nOutlier detection:")
            for feature in self.feature_names + ['recovery_score']:
                outliers = identify_outliers(merged_df, feature)
                if len(outliers) > 0:
                    print(f"- {feature}: {len(outliers)} outliers detected")
        
        # Create feature matrix 
        features = merged_df[self.feature_names].copy()
        
        # Check values against physiological bounds
        if verbose:
            print("\nData validation against physiological bounds:")
            for feature, (lower, upper) in self.feature_bounds.items():
                if feature in features.columns:
                    out_of_bounds = merged_df[(merged_df[feature] < lower) | (merged_df[feature] > upper)]
                    if len(out_of_bounds) > 0:
                        print(f"- {feature}: {len(out_of_bounds)} values outside expected range [{lower}-{upper}]")
        
        # Cap values to physiological bounds
        for feature, (lower, upper) in self.feature_bounds.items():
            if feature in features.columns:
                features[feature] = features[feature].clip(lower, upper)
        
        # Extract target variable 
        target = merged_df['recovery_score']
        
        # Ensure recovery score is within expected range (0-100)
        target = target.clip(0, 100)
        
        if verbose:
            print(f"\nFinal dataset: {len(features)} records ready for modeling")
        
        # Return both features, target, and the merged dataframe with dates
        return features, target, merged_df
    
    def train(self, save_model=True):
        """
        Train the Ridge regression model.
        
        Args:
            save_model (bool): Whether to save the trained model
            
        Returns:
            dict: Model performance metrics
        """
        if self.verbose:
            print("\nTraining the Ridge regression model...")
        
        # Load and prepare data
        X, y, merged_df = self.load_data(verbose=self.verbose)
        
        # Store data for later use
        self.data = merged_df
        
        # First, train a model using cross-validation for visualization
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        all_predictions = []
        all_actual = []
        all_indices = []
        
        # For each fold, train a model and make predictions
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create and train a new model
            fold_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=1.0))
            ])
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = fold_model.predict(X_test)
            
            # Save results
            all_predictions.extend(y_pred)
            all_actual.extend(y_test)
            all_indices.extend(test_idx)
        
        # Create a DataFrame of all CV results
        cv_results = pd.DataFrame({
            'actual': all_actual,
            'predicted': all_predictions,
            'index': all_indices
        })
        cv_results = cv_results.sort_values('index')
        
        # Calculate CV metrics
        cv_r2 = r2_score(cv_results['actual'], cv_results['predicted'])
        cv_rmse = np.sqrt(mean_squared_error(cv_results['actual'], cv_results['predicted']))
        cv_mae = mean_absolute_error(cv_results['actual'], cv_results['predicted'])
        
        # Now train on full dataset
        self.model.fit(X, y)
        
        # Get feature importance
        feature_importance = self.model.named_steps['regressor'].coef_
        
        # Store results
        self.metrics = {
            'cv_r2': cv_r2,
            'cv_rmse': cv_rmse,
            'cv_mae': cv_mae,
            'feature_importance': feature_importance,
        }
        
        # Store CV data for later visualization
        self.cv_data = {
            'X': X,
            'y': y,
            'merged_df': merged_df,
            'cv_results': cv_results
        }
        
        if self.verbose:
            print("\nRidge Regression Model Performance:")
            print(f"Cross-validation R²: {cv_r2:.4f}")
            print(f"Cross-validation RMSE: {cv_rmse:.4f}")
            print(f"Cross-validation MAE: {cv_mae:.4f}")
            
            print("\nFeature Importance (Coefficients):")
            for feature, importance in zip(self.feature_names, feature_importance):
                print(f"{feature}: {importance:.4f}")
        
        # Save the trained model if requested
        if save_model:
            self.save_model(MODEL_FILE)
            
        return self.metrics
    
    def save_model(self, model_path=None):
        """
        Save the trained model to a file.
        
        Args:
            model_path (str): Path to save the model. If None, uses the default path.
            
        Returns:
            str: Path where the model was saved
        """
        if model_path is None:
            model_path = MODEL_FILE
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model, feature names, and bounds
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_bounds': self.feature_bounds,
            'metrics': getattr(self, 'metrics', None)
        }
        
        joblib.dump(model_data, model_path)
        
        if self.verbose:
            print(f"Model saved to: {model_path}")
            
        return model_path
    
    def load_model(self, model_path=None):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model file. If None, uses the default path.
            
        Returns:
            bool: True if loaded successfully
        """
        if model_path is None:
            model_path = MODEL_FILE
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model and metadata
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_bounds = model_data['feature_bounds']
        
        if 'metrics' in model_data and model_data['metrics'] is not None:
            self.metrics = model_data['metrics']
            
        if self.verbose:
            print(f"Model loaded from: {model_path}")
            
        return True
    
    def predict(self, garmin_data):
        """
        Predict Whoop recovery score from Garmin data.
        
        Args:
            garmin_data (dict or DataFrame): Garmin data with the required features.
            
        Returns:
            float: Predicted Whoop recovery score.
        """
        # Convert dict to DataFrame if necessary
        if isinstance(garmin_data, dict):
            garmin_data = pd.DataFrame([garmin_data])
        
        # Ensure all required features are present
        missing_features = [f for f in self.feature_names if f not in garmin_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Handle missing values 
        input_data = garmin_data[self.feature_names].copy()
        input_data = input_data.fillna(input_data.median())
        
        # Clip values to physiological bounds
        for feature in self.feature_names:
            if feature in self.feature_bounds:
                lower, upper = self.feature_bounds[feature]
                input_data[feature] = input_data[feature].clip(lower, upper)
        
        # Make prediction
        prediction = self.model.predict(input_data)
        
        # Ensure prediction is within expected range
        prediction = np.clip(prediction, 0, 100)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def visualize_feature_importance(self):
        """
        Visualize feature importance for Ridge regression.
        
        Returns:
            matplotlib.figure.Figure: The feature importance plot figure.
        """
        if not hasattr(self, 'metrics') or 'feature_importance' not in self.metrics:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importances (coefficients)
        importances = self.metrics['feature_importance']
        
        # Convert to absolute values for sorting
        abs_importances = np.abs(importances)
        
        # Create sorted indices based on absolute importance
        indices = np.argsort(abs_importances)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create a color map: blue for negative, red for positive coefficients
        colors = ['blue' if coef < 0 else 'red' for coef in importances[indices]]
        
        # Create the horizontal bar chart
        bars = plt.barh(range(len(indices)), abs_importances[indices], align='center', color=colors)
        
        # Add actual coefficient values as text labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{importances[indices[i]]:.2f}', 
                    va='center', fontsize=10)
        
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Absolute Coefficient Value')
        plt.title('Ridge Regression Coefficients for Whoop Recovery Score Prediction')
        
        # Add a legend
        import matplotlib.patches as mpatches
        pos_patch = mpatches.Patch(color='red', label='Positive (Increases Recovery)')
        neg_patch = mpatches.Patch(color='blue', label='Negative (Decreases Recovery)')
        plt.legend(handles=[pos_patch, neg_patch], loc='lower right')
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_predictions(self):
        """
        Visualize actual vs predicted recovery scores using cross-validation results.
        
        Returns:
            matplotlib.figure.Figure: The visualization plot figure.
        """
        if not hasattr(self, 'cv_data'):
            raise ValueError("Model not trained with cross-validation. Call train() first.")
        
        # Extract data from cross-validation
        cv_results = self.cv_data['cv_results']
        
        # Create scatter plot
        plt.figure(figsize=(10, 7))
        plt.scatter(cv_results['actual'], cv_results['predicted'], alpha=0.6, s=80, c='steelblue')
        
        # Add identity line
        min_val = min(min(cv_results['actual']), min(cv_results['predicted'])) - 5
        max_val = max(max(cv_results['actual']), max(cv_results['predicted'])) + 5
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        # Add regression line
        z = np.polyfit(cv_results['actual'], cv_results['predicted'], 1)
        p = np.poly1d(z)
        plt.plot(np.sort(cv_results['actual']), p(np.sort(cv_results['actual'])), "r-", lw=2, 
                 label=f"Regression Line (y = {z[0]:.2f}x + {z[1]:.2f})")
        
        # Annotate with metrics
        metrics_text = f"R² = {self.metrics['cv_r2']:.3f}\nRMSE = {self.metrics['cv_rmse']:.2f}\nMAE = {self.metrics['cv_mae']:.2f}"
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='left', va='top', fontsize=12)
        
        # Add labels and title
        plt.xlabel('Actual Recovery Score', fontsize=12)
        plt.ylabel('Predicted Recovery Score', fontsize=12)
        plt.title('Ridge Regression: Actual vs Predicted Whoop Recovery Scores (Cross-Validation)', fontsize=14)
        
        # Add x=y line explanation
        plt.legend(loc='lower right')
        plt.text(min_val + 5, max_val - 10, "Perfect Prediction (x=y)", 
                fontsize=10, rotation=38, rotation_mode='anchor',
                transform_rotates_text=True)
        
        # Set equal aspect ratio
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_prediction_over_time(self):
        """
        Visualize predictions over time using cross-validation results.
        
        Returns:
            matplotlib.figure.Figure: The time series plot figure.
        """
        if not hasattr(self, 'cv_data'):
            raise ValueError("Model not trained with cross-validation. Call train() first.")
        
        # Extract data from cross-validation
        cv_results = self.cv_data['cv_results']
        merged_df = self.cv_data['merged_df']
        
        # Add dates to the results
        dates = merged_df.iloc[cv_results['index']]['date'].values
        
        # Create a DataFrame with dates, actual and predicted values
        results_df = pd.DataFrame({
            'date': dates,
            'actual': cv_results['actual'],
            'predicted': cv_results['predicted'],
            'error': cv_results['actual'] - cv_results['predicted']
        }).sort_values('date')
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot actual and predicted values
        ax1.plot(results_df['date'], results_df['actual'], 'o-', label='Actual', color='green', markersize=8)
        ax1.plot(results_df['date'], results_df['predicted'], 's--', label='Predicted', color='blue', markersize=6)
        
        # Fill the area between actual and predicted
        ax1.fill_between(results_df['date'], results_df['actual'], results_df['predicted'], 
                        alpha=0.2, color='gray')
        
        # Add a horizontal line at 50 (mid-point of recovery scale)
        ax1.axhline(y=50, color='red', linestyle=':', alpha=0.7, label='Recovery Threshold (50)')
        
        # Format the plot
        ax1.set_title('Whoop Recovery Score: Actual vs Predicted Over Time (Cross-Validation)', fontsize=14)
        ax1.set_ylabel('Recovery Score', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Add error bars in the second subplot
        ax2.bar(results_df['date'], results_df['error'], color='darkred', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format the error subplot
        ax2.set_title('Prediction Error (Actual - Predicted)', fontsize=12)
        ax2.set_ylabel('Error', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.xticks(rotation=45)
        
        # Add a line showing average error
        avg_error = results_df['error'].mean()
        ax2.axhline(y=avg_error, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Avg Error: {avg_error:.2f}')
        ax2.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def visualize_feature_relationships(self):
        """
        Create a pairplot showing relationships between features and target.
        
        Returns:
            matplotlib.figure.Figure: The correlation plot figure.
        """
        # Load data
        X, y, _ = self.load_data(verbose=False)
        
        if len(X) == 0:
            print("No data available for visualization")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, "No data available for visualization", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Combine features and target for visualization
        data = X.copy()
        data['recovery_score'] = y
        
        # Set plot style
        sns.set(style="ticks")
        
        # Create the pairplot
        g = sns.pairplot(data, height=2.5, corner=True, 
                         plot_kws={"s": 60, "alpha": 0.6})
        
        # Enhance the plot
        g.fig.suptitle("Relationships Between Features and Recovery Score", 
                      fontsize=16, y=1.02)
        
        # Create a correlation heatmap
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")
        
        plt.title("Feature Correlation Heatmap", fontsize=14)
        plt.tight_layout()
        
        # Return the pairplot
        return g.fig

def main():
    """Main function to demonstrate model training and prediction."""
    # Create an instance of the predictor
    predictor = WhoopRecoveryPredictor(verbose=True)
    
    # Create a folder for visualizations if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Train the model and get performance metrics
    print("\nTraining Ridge regression model...")
    metrics = predictor.train(save_model=True)
    
    # Create and save visualizations
    print("\nGenerating visualizations...")
    
    # Feature importance plot
    fig1 = predictor.visualize_feature_importance()
    fig1.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png'))
    
    # Predictions plot
    fig2 = predictor.visualize_predictions()
    fig2.savefig(os.path.join(VISUALIZATIONS_DIR, 'actual_vs_predicted.png'))
    
    # Time series plot
    fig3 = predictor.visualize_prediction_over_time()
    fig3.savefig(os.path.join(VISUALIZATIONS_DIR, 'predictions_over_time.png'))
    
    # Feature relationships
    fig4 = predictor.visualize_feature_relationships()
    fig4.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_relationships.png'))
    
    print(f"\nVisualizations saved to: {VISUALIZATIONS_DIR}")
    
    # Example prediction
    print("\nExample prediction:")
    # Get sample data
    X, _, _ = predictor.load_data(verbose=False)
    if len(X) > 0:
        sample_data = X.iloc[0].to_dict()
        
        print("Sample Garmin data:")
        for feature, value in sample_data.items():
            print(f"  {feature}: {value}")
        
        prediction = predictor.predict(sample_data)
        print(f"\nPredicted Whoop recovery score: {prediction:.2f}")
    else:
        print("No data available for prediction example")
    
    print("\nModel saved and ready to use from other code.")
    print("To load the model in another script:")
    print("from analyses.predict_whoop_recovery import WhoopRecoveryPredictor")
    print("predictor = WhoopRecoveryPredictor(verbose=False)")
    print("predictor.load_model()")
    print("recovery_score = predictor.predict(garmin_data)")

if __name__ == "__main__":
    main() 