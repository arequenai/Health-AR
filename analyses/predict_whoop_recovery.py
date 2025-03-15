import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

# Import config from ETL package if available
try:
    from etl import config
    # If imported from config, make sure paths are absolute
    GARMIN_DAILY_FILE = os.path.join(PROJECT_ROOT, 'data', 'garmin_daily.csv')
    WHOOP_FILE = os.path.join(PROJECT_ROOT, 'data', 'whoop.csv')
except ImportError:
    # Default paths if config module is not available
    GARMIN_DAILY_FILE = os.path.join(PROJECT_ROOT, 'data', 'garmin_daily.csv')
    WHOOP_FILE = os.path.join(PROJECT_ROOT, 'data', 'whoop.csv')

print(f"Using data files: \n- Garmin: {GARMIN_DAILY_FILE}\n- Whoop: {WHOOP_FILE}")
print(f"Visualizations will be saved to: {VISUALIZATIONS_DIR}")

class WhoopRecoveryPredictor:
    """
    A class to predict Whoop recovery scores using Garmin data.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor.
        
        Args:
            model_path (str, optional): Path to a saved model file. If provided, loads the model.
        """
        self.model = None
        self.feature_names = [
            'restingHeartRate',
            'sleepingSeconds',
            'sleep_score'
        ]
        
        # Physiological bounds for data validation
        self.feature_bounds = {
            'restingHeartRate': (30, 120),              # Reasonable HR range
            'sleepingSeconds': (0, 12*3600),            # 0 to 12 hours in seconds
            'sleep_score': (0, 100)                     # Sleep score 0-100
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_data(self, verbose=True):
        """
        Load and preprocess data from Garmin and Whoop.
        
        Args:
            verbose (bool): Whether to print data quality information
            
        Returns:
            tuple: (X, y) features and target for training
        """
        # Load data
        df_garmin = pd.read_csv(GARMIN_DAILY_FILE)
        df_whoop = pd.read_csv(WHOOP_FILE)
        
        if verbose:
            print(f"\nData Preprocessing Report:")
            print(f"- Garmin records: {len(df_garmin)}")
            print(f"- Whoop records: {len(df_whoop)}")
        
        # Convert date columns to datetime for proper merging
        df_garmin['date'] = pd.to_datetime(df_garmin['date'])
        df_whoop['date'] = pd.to_datetime(df_whoop['date'])
        
        # Check for missing dates in each dataset
        garmin_dates = set(df_garmin['date'].dt.strftime('%Y-%m-%d'))
        whoop_dates = set(df_whoop['date'].dt.strftime('%Y-%m-%d'))
        
        # Find dates in one dataset but not the other
        dates_only_in_garmin = garmin_dates - whoop_dates
        dates_only_in_whoop = whoop_dates - garmin_dates
        
        if verbose and (dates_only_in_garmin or dates_only_in_whoop):
            print(f"- Warning: {len(dates_only_in_garmin)} dates in Garmin but not in Whoop")
            print(f"- Warning: {len(dates_only_in_whoop)} dates in Whoop but not in Garmin")
        
        # Merge datasets on date (inner join keeps only dates present in both datasets)
        merged_df = pd.merge(df_garmin, df_whoop[['date', 'recovery_score']], on='date', how='inner')
        
        if verbose:
            print(f"- After merge: {len(merged_df)} records")
        
        # Convert sleepingSeconds to hours for better interpretability
        merged_df['sleepingHours'] = merged_df['sleepingSeconds'] / 3600
        
        # Filter out days with zero or very minimal sleep time (less than 1 minute)
        zero_sleep_count = len(merged_df[merged_df['sleepingSeconds'] < 60])
        if zero_sleep_count > 0:
            if verbose:
                print(f"- Filtering out {zero_sleep_count} days with little or no sleep recorded (< 1 minute)")
            merged_df = merged_df[merged_df['sleepingSeconds'] >= 60]
        
        # Check for missing values before cleaning
        missing_values = merged_df[self.feature_names + ['recovery_score']].isnull().sum()
        if verbose and missing_values.sum() > 0:
            print("\nMissing values before cleaning:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"- {col}: {count} missing values")
        
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
                out_of_bounds = merged_df[(merged_df[feature] < lower) | (merged_df[feature] > upper)]
                if len(out_of_bounds) > 0:
                    print(f"- {feature}: {len(out_of_bounds)} values outside expected range [{lower}-{upper}]")
        
        # Handle missing values
        features = features.fillna(features.median())  # Using median instead of mean as it's more robust to outliers
        
        # Extract target variable 
        target = merged_df['recovery_score']
        
        # Check if target has missing values
        missing_target = target.isnull().sum()
        if missing_target > 0:
            if verbose:
                print(f"\nWarning: {missing_target} missing recovery scores - rows will be dropped")
            # Drop rows with missing target values
            valid_indices = target.notna()
            features = features[valid_indices]
            target = target[valid_indices]
        
        # Cap values to physiological bounds
        for feature, (lower, upper) in self.feature_bounds.items():
            features[feature] = features[feature].clip(lower, upper)
        
        # Ensure recovery score is within expected range (0-100)
        target = target.clip(0, 100)
        
        if verbose:
            print(f"\nFinal dataset: {len(features)} records ready for modeling")
        
        # Return both features, target, and the merged dataframe with dates
        return features, target, merged_df[valid_indices] if missing_target > 0 else merged_df
    
    def train(self, save_path=None, verbose=True):
        """
        Train a model to predict Whoop recovery scores.
        
        Args:
            save_path (str, optional): Path to save the trained model.
            verbose (bool): Whether to print detailed information
            
        Returns:
            dict: Dictionary containing model performance metrics.
        """
        # Load and prepare data
        X, y, _ = self.load_data(verbose=verbose)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if verbose:
            print(f"\nTraining set: {X_train.shape[0]} samples")
            print(f"Testing set: {X_test.shape[0]} samples")
            
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Get feature importances
        importances = self.model.named_steps['regressor'].feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))
        
        # Save model if path is provided
        if save_path:
            # Only create directory if the save_path contains a directory component
            directory = os.path.dirname(save_path)
            if directory:  # Only try to create directory if it's not empty
                os.makedirs(directory, exist_ok=True)
            joblib.dump(self.model, save_path)
        
        # Return performance metrics
        metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'feature_importance': feature_importance
        }
        
        return metrics
    
    def predict(self, garmin_data):
        """
        Predict Whoop recovery score from Garmin data.
        
        Args:
            garmin_data (dict or DataFrame): Garmin data with the required features.
            
        Returns:
            float: Predicted Whoop recovery score.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Convert dict to DataFrame if necessary
        if isinstance(garmin_data, dict):
            garmin_data = pd.DataFrame([garmin_data])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in garmin_data.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Handle missing values 
        input_data = garmin_data[self.feature_names].copy()
        input_data = input_data.fillna(input_data.median())
        
        # Clip values to physiological bounds
        for feature, (lower, upper) in self.feature_bounds.items():
            input_data[feature] = input_data[feature].clip(lower, upper)
        
        # Make prediction
        prediction = self.model.predict(input_data)
        
        # Ensure prediction is within expected range
        prediction = np.clip(prediction, 0, 100)
        
        return prediction[0]
    
    def load_model(self, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
    
    def visualize_feature_importance(self):
        """
        Visualize feature importance.
        
        Returns:
            matplotlib.figure.Figure: The feature importance plot figure.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Get feature importances
        importances = self.model.named_steps['regressor'].feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance for Whoop Recovery Score Prediction')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_predictions(self):
        """
        Visualize actual vs predicted recovery scores.
        
        Returns:
            matplotlib.figure.Figure: The visualization plot figure.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Load data
        X, y, _ = self.load_data(verbose=False)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Create DataFrame for visualization
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        })
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual Recovery Score')
        plt.ylabel('Predicted Recovery Score')
        plt.title('Actual vs Predicted Whoop Recovery Scores')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_data_distribution(self):
        """
        Visualize the distribution of features and target.
        
        Returns:
            matplotlib.figure.Figure: The distribution plot figure.
        """
        # Load data
        X, y, _ = self.load_data(verbose=False)
        
        # Combine features and target for plotting
        data = X.copy()
        data['recovery_score'] = y
        
        # Create distribution plots
        fig, axs = plt.subplots(len(self.feature_names) + 1, 1, figsize=(12, 3 * (len(self.feature_names) + 1)))
        
        for i, col in enumerate(self.feature_names + ['recovery_score']):
            sns.histplot(data[col], kde=True, ax=axs[i])
            axs[i].set_title(f'Distribution of {col}')
            axs[i].set_xlabel(col)
            
        plt.tight_layout()
        return fig
    
    def visualize_feature_correlations(self):
        """
        Visualize the correlation between each feature and the Whoop recovery score.
        
        Returns:
            matplotlib.figure.Figure: The correlation plot figure.
        """
        # Load data
        X, y, _ = self.load_data(verbose=False)
        
        # Combine features and target for correlation calculation
        data = X.copy()
        data['recovery_score'] = y
        
        # Calculate correlations with the target
        correlations = {}
        for feature in self.feature_names:
            corr = data[feature].corr(data['recovery_score'])
            correlations[feature] = corr
            
        # Create subplots for each feature
        fig, axs = plt.subplots(len(self.feature_names), 1, figsize=(12, 4 * len(self.feature_names)))
        
        # If there's only one feature, axs won't be an array
        if len(self.feature_names) == 1:
            axs = [axs]
            
        for i, feature in enumerate(self.feature_names):
            # Create scatter plot with regression line
            ax = axs[i]
            sns.regplot(x=feature, y='recovery_score', data=data, ax=ax, scatter_kws={'alpha': 0.5})
            
            # Get correlation coefficient and add to plot
            corr = correlations[feature]
            ax.set_title(f'Correlation: {feature} vs recovery_score (r = {corr:.3f})')
            ax.set_xlabel(feature)
            ax.set_ylabel('Whoop Recovery Score')
            
            # Add text annotation with correlation value
            if abs(corr) > 0.5:
                strength = "Strong"
            elif abs(corr) > 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"
                
            direction = "Positive" if corr > 0 else "Negative"
            ax.text(0.05, 0.95, f"{direction} {strength} Correlation", 
                   transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        plt.tight_layout()
        return fig
    
    def identify_largest_deviations(self, n=10):
        """
        Identify and visualize days with the largest deviations between actual and predicted recovery scores.
        
        Args:
            n (int): Number of days to display (default: 10)
            
        Returns:
            tuple: (DataFrame with top deviations, matplotlib.figure.Figure)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Load data including dates
        X, y, merged_df = self.load_data(verbose=False)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Create DataFrame with dates, actual and predicted values
        results_df = pd.DataFrame({
            'date': merged_df['date'],
            'actual': y,
            'predicted': y_pred,
            'deviation': np.abs(y - y_pred)
        })
        
        # Sort by deviation (descending) and get top n
        top_deviations = results_df.sort_values('deviation', ascending=False).head(n)
        
        # Format the date for better display
        top_deviations['date'] = top_deviations['date'].dt.strftime('%Y-%m-%d')
        
        # Create a bar plot to visualize the top deviations
        plt.figure(figsize=(12, 8))
        
        # Plot actual vs predicted for top deviations
        dates = top_deviations['date'].values
        x = np.arange(len(dates))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        actual_bars = ax.bar(x - width/2, top_deviations['actual'], width, label='Actual', color='steelblue')
        predicted_bars = ax.bar(x + width/2, top_deviations['predicted'], width, label='Predicted', color='indianred')
        
        # Add a secondary axis for deviation
        ax2 = ax.twinx()
        ax2.plot(x, top_deviations['deviation'], 'go-', linewidth=2, markersize=8, label='Deviation')
        ax2.set_ylabel('Absolute Deviation', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Add labels, title and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Recovery Score')
        ax.set_title('Top Days with Largest Deviation Between Actual and Predicted Recovery Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        
        # Add values above bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        add_labels(actual_bars)
        add_labels(predicted_bars)
        
        # Add legends for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        return top_deviations, fig

def main():
    """Main function to demonstrate model training and prediction."""
    # Create an instance of the predictor
    predictor = WhoopRecoveryPredictor()
    
    # Create a folder for visualizations if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Generate data distribution visualization
    print("\nGenerating data distribution visualization...")
    fig_dist = predictor.visualize_data_distribution()
    fig_dist.savefig(os.path.join(VISUALIZATIONS_DIR, 'data_distribution.png'))
    
    # Generate feature correlation visualization
    print("\nGenerating feature correlation visualization...")
    fig_corr = predictor.visualize_feature_correlations()
    fig_corr.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_correlations.png'))
    
    # Train the model and get performance metrics
    print("\nTraining model...")
    metrics = predictor.train(save_path=os.path.join(ANALYSES_DIR, 'whoop_recovery_model.joblib'), verbose=True)
    
    # Print performance metrics
    print("\nModel Performance:")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Testing R²: {metrics['test_r2']:.4f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    
    print("\nFeature Importance:")
    for feature, importance in sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Create and save visualizations
    print("\nGenerating visualizations...")
    
    # Feature importance plot
    fig1 = predictor.visualize_feature_importance()
    fig1.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png'))
    
    # Predictions plot
    fig2 = predictor.visualize_predictions()
    fig2.savefig(os.path.join(VISUALIZATIONS_DIR, 'actual_vs_predicted.png'))
    
    # Identify and visualize days with largest deviations
    print("\nIdentifying days with largest deviations...")
    top_deviations, fig_dev = predictor.identify_largest_deviations(n=10)
    fig_dev.savefig(os.path.join(VISUALIZATIONS_DIR, 'largest_deviations.png'))
    
    # Print the top deviation days
    print("\nTop 10 days with largest deviations between actual and predicted recovery scores:")
    for i, (_, row) in enumerate(top_deviations.iterrows(), 1):
        print(f"{i}. Date: {row['date']} - Actual: {row['actual']:.1f}, Predicted: {row['predicted']:.1f}, Deviation: {row['deviation']:.1f}")
    
    print(f"\nVisualizations saved to: {VISUALIZATIONS_DIR}")
    
    # Example prediction
    print("\nExample prediction:")
    # Get sample data
    X, _, _ = predictor.load_data(verbose=False)
    sample_data = X.iloc[0].to_dict()
    
    print("Sample Garmin data:")
    for feature, value in sample_data.items():
        print(f"  {feature}: {value}")
    
    prediction = predictor.predict(sample_data)
    print(f"\nPredicted Whoop recovery score: {prediction:.2f}")

if __name__ == "__main__":
    main() 