import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    Automated pipeline for real-time fraud detection
    """
    
    def __init__(self, model_path='../models/'):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.threshold = 0.5
        self.load_pipeline()
    
    def load_pipeline(self):
        """Load trained models and preprocessing components"""
        try:
            # Load best model (adjust filename based on your best model)
            self.model = joblib.load(self.model_path / 'best_model_random_forest.joblib')
            print("✓ Model loaded successfully")
            
            # Load preprocessing components
            self.scaler = joblib.load(self.model_path / 'scaler.pkl')
            self.feature_selector = joblib.load(self.model_path / 'feature_selector.pkl')
            print("✓ Preprocessing components loaded")
            
            # Load feature names from training
            with open(self.model_path / 'feature_names.json', 'r') as f:
                self.feature_names = json.load(f)
            print("✓ Feature names loaded")
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise
    
    def create_customer_features(self, invoice_data, client_data):
        """
        Create features from raw invoice and client data
        Same feature engineering as training
        """
        # Merge invoice and client data
        merged_data = invoice_data.merge(client_data, on='client_id', how='left')
        
        # Create customer-level features
        features = merged_data.groupby('client_id').agg({
            'consommation_level_1': ['mean', 'std', 'max', 'min', 'sum'],
            'consommation_level_2': ['mean', 'std', 'max', 'min', 'sum'],
            'consommation_level_3': ['mean', 'std', 'max', 'min', 'sum'],
            'consommation_level_4': ['mean', 'std', 'max', 'min', 'sum'],
            'new_index': ['count'],
            'months_number': ['mean', 'std', 'min', 'max'],
            'counter_coefficient': ['mean', 'std'],
            'reading_remarque': ['nunique'],
            'tarif_type': ['nunique']
        }).fillna(0)
        
        # Flatten column names
        features.columns = ['_'.join(col).strip() for col in features.columns]
        
        # Add client demographic features
        client_features = client_data.set_index('client_id')[['client_catg', 'region', 'district']]
        features = features.join(client_features, how='left')
        
        # Calculate additional fraud indicators
        features['consumption_volatility'] = merged_data.groupby('client_id')['consommation_level_1'].std()
        features['avg_monthly_consumption'] = merged_data.groupby('client_id')['consommation_level_1'].mean()
        features['billing_frequency'] = merged_data.groupby('client_id').size()
        
        # Handle missing values
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def preprocess_features(self, features_df):
        """Apply same preprocessing as training"""
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            for feature in missing_features:
                features_df[feature] = 0
        
        # Select only required features in correct order
        features_df = features_df[self.feature_names]
        
        # Apply feature selection
        features_selected = self.feature_selector.transform(features_df)
        
        # Apply scaling
        features_scaled = self.scaler.transform(features_selected)
        
        return features_scaled
    
    def predict_fraud(self, invoice_data, client_data):
        """
        Predict fraud for new customer data
        
        Args:
            invoice_data: DataFrame with invoice records
            client_data: DataFrame with client information
            
        Returns:
            DataFrame with predictions
        """
        # Create features
        features = self.create_customer_features(invoice_data, client_data)
        
        # Preprocess features
        features_processed = self.preprocess_features(features)
        
        # Get predictions
        fraud_probabilities = self.model.predict_proba(features_processed)[:, 1]
        fraud_predictions = (fraud_probabilities > self.threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'client_id': features.index,
            'fraud_probability': fraud_probabilities,
            'fraud_prediction': fraud_predictions,
            'risk_level': self.classify_risk(fraud_probabilities),
            'timestamp': datetime.now()
        })
        
        return results
    
    def classify_risk(self, probabilities):
        """Classify risk levels based on probability"""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append('LOW')
            elif prob < 0.7:
                risk_levels.append('MEDIUM')
            else:
                risk_levels.append('HIGH')
        return risk_levels
    
    def batch_process(self, data_path, output_path):
        """Process batch of customers"""
        print(f"Processing batch from {data_path}")
        
        # Load data
        invoice_data = pd.read_csv(data_path / 'invoice_data.csv')
        client_data = pd.read_csv(data_path / 'client_data.csv')
        
        # Get predictions
        results = self.predict_fraud(invoice_data, client_data)
        
        # Save results
        results.to_csv(output_path / f'fraud_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', 
                      index=False)
        
        # Generate summary report
        summary = {
            'total_customers': len(results),
            'high_risk_customers': len(results[results['risk_level'] == 'HIGH']),
            'medium_risk_customers': len(results[results['risk_level'] == 'MEDIUM']),
            'low_risk_customers': len(results[results['risk_level'] == 'LOW']),
            'avg_fraud_probability': results['fraud_probability'].mean(),
            'max_fraud_probability': results['fraud_probability'].max(),
            'processing_time': datetime.now().isoformat()
        }
        
        with open(output_path / 'batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Processed {len(results)} customers")
        print(f"✓ High risk: {summary['high_risk_customers']}")
        print(f"✓ Results saved to {output_path}")
        
        return results, summary

class PipelineMonitor:
    """Monitor pipeline performance and data drift"""
    
    def __init__(self, results_path='../results/'):
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
    def log_predictions(self, predictions, batch_id=None):
        """Log predictions for monitoring"""
        log_entry = {
            'batch_id': batch_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'fraud_rate': predictions['fraud_prediction'].mean(),
            'avg_probability': predictions['fraud_probability'].mean(),
            'high_risk_count': len(predictions[predictions['risk_level'] == 'HIGH'])
        }
        
        # Append to monitoring log
        log_file = self.results_path / 'monitoring_log.json'
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"✓ Logged predictions: {log_entry}")
    
    def check_drift(self, current_predictions, baseline_fraud_rate=0.05):
        """Simple drift detection"""
        current_fraud_rate = current_predictions['fraud_prediction'].mean()
        
        if abs(current_fraud_rate - baseline_fraud_rate) > 0.02:  # 2% threshold
            alert = {
                'type': 'DRIFT_ALERT',
                'timestamp': datetime.now().isoformat(),
                'current_fraud_rate': current_fraud_rate,
                'baseline_fraud_rate': baseline_fraud_rate,
                'difference': abs(current_fraud_rate - baseline_fraud_rate)
            }
            
            print(f"⚠️  DRIFT ALERT: Fraud rate changed from {baseline_fraud_rate:.3f} to {current_fraud_rate:.3f}")
            
            # Save alert
            with open(self.results_path / 'drift_alerts.json', 'a') as f:
                json.dump(alert, f)
                f.write('\n')
            
            return True
        
        return False

def save_feature_names(model_path, feature_names):
    """Save feature names for pipeline loading"""
    with open(Path(model_path) / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print("✓ Feature names saved")

def main():
    """Demo the automated pipeline"""
    print("=== AUTOMATED FRAUD DETECTION PIPELINE ===\n")
    
    # Initialize pipeline
    try:
        pipeline = FraudDetectionPipeline()
        monitor = PipelineMonitor()
        
        print("✓ Pipeline initialized successfully")
        
        # Example: Process new data (you would replace this with actual data paths)
        print("\n--- Example: Processing new customer data ---")
        
        # Create sample data for demo
        sample_invoice = pd.DataFrame({
            'client_id': ['new_client_1', 'new_client_2'],
            'consommation_level_1': [150, 50],  # Suspicious low consumption
            'consommation_level_2': [0, 0],
            'consommation_level_3': [0, 0],
            'consommation_level_4': [0, 0],
            'new_index': [1500, 500],
            'months_number': [1, 1],
            'counter_coefficient': [1, 1],
            'reading_remarque': [1, 2],
            'tarif_type': [11, 11]
        })
        
        sample_client = pd.DataFrame({
            'client_id': ['new_client_1', 'new_client_2'],
            'client_catg': [11, 11],
            'region': [101, 102],
            'district': [60, 61]
        })
        
        # Get predictions
        predictions = pipeline.predict_fraud(sample_invoice, sample_client)
        print("\nPredictions:")
        print(predictions)
        
        # Monitor predictions
        monitor.log_predictions(predictions)
        monitor.check_drift(predictions)
        
        print("\n✓ Pipeline demo completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure you have trained models in ../models/ directory")

if __name__ == "__main__":
    main()