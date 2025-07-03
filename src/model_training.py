import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest, f_classif

# XGBoost
import xgboost as xgb

class FraudModelTrainer:
    """
    Enhanced train and evaluate ML models for energy fraud detection
    Focus on Random Forest and XGBoost with improved techniques for imbalanced data
    """
    
    def _init_(self, data_path='../data/', model_path='../models/', results_path='../results/'):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.results_path = Path(results_path)
        
        # Create directories if they don't exist
        self.model_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.scaler = None
        self.feature_selector = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load features and prepare train/test split"""
        print("Loading customer features...")
        
        # Load the engineered features
        features_df = pd.read_csv(self.data_path / 'customer_features.csv', index_col=0)
        
        print(f"✓ Loaded features: {features_df.shape}")
        print(f"✓ Features: {features_df.shape[1]} columns")
        
        # Separate features and target
        if 'target' not in features_df.columns:
            raise ValueError("Target column not found in features!")
        
        X = features_df.drop('target', axis=1)
        y = features_df['target']
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")
        print(f"✓ Fraud rate: {y.mean():.4f}")
        
        # Handle any remaining missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Convert non-numeric columns to numeric
        print("Converting non-numeric columns...")
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try to convert to datetime first, then to numeric
                    X[col] = pd.to_datetime(X[col], errors='coerce')
                    X[col] = (X[col] - pd.Timestamp('1970-01-01')).dt.days
                except:
                    # If datetime conversion fails, try direct numeric conversion
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Fill any NaN values created during conversion
                X[col] = X[col].fillna(0)
        
        # Remove constant features (they don't help with classification)
        constant_features = X.columns[X.std() == 0]
        if len(constant_features) > 0:
            print(f"Removing {len(constant_features)} constant features: {list(constant_features)}")
            X = X.drop(columns=constant_features)
        
        # Train/test split - stratified to maintain fraud ratio
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        print(f"✓ Train fraud rate: {self.y_train.mean():.4f}")
        print(f"✓ Test fraud rate: {self.y_test.mean():.4f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features using RobustScaler (better for outliers)"""
        print("Scaling features with RobustScaler...")
        
        # Use RobustScaler instead of StandardScaler - more robust to outliers
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Features scaled")
        
        # Save scaler
        joblib.dump(self.scaler, self.model_path / 'scaler.pkl')
        print("✓ Scaler saved")
    
    def feature_selection(self):
        """Select most important features"""
        print("Performing feature selection...")
        
        # Select top features based on statistical tests
        k_features = min(50, self.X_train.shape[1])  # Select top 50 features or all if less
        self.feature_selector = SelectKBest(f_classif, k=k_features)
        
        X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.feature_selector.transform(self.X_test)
        
        selected_features = self.X_train.columns[self.feature_selector.get_support()]
        print(f"✓ Selected {len(selected_features)} features out of {self.X_train.shape[1]}")
        
        # Update training data with selected features
        self.X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=self.X_test.index)
        
        # Save feature selector
        joblib.dump(self.feature_selector, self.model_path / 'feature_selector.pkl')
        print("✓ Feature selector saved")
        
        return selected_features
    
    def apply_resampling(self, strategy='smote'):
        """Apply resampling to handle class imbalance"""
        print(f"Applying {strategy} resampling...")
        
        if strategy == 'smote':
            # SMOTE - Synthetic Minority Oversampling
            resampler = SMOTE(random_state=42, k_neighbors=3)
        elif strategy == 'smoteenn':
            # SMOTE + Edited Nearest Neighbours
            resampler = SMOTEENN(random_state=42)
        elif strategy == 'undersampling':
            # Random undersampling
            resampler = RandomUnderSampler(random_state=42)
        else:
            print("No resampling applied")
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()
            return
        
        self.X_train_resampled, self.y_train_resampled = resampler.fit_resample(
            self.X_train, self.y_train
        )
        
        print(f"✓ Original training set: {self.X_train.shape[0]} samples")
        print(f"✓ Resampled training set: {self.X_train_resampled.shape[0]} samples")
        print(f"✓ Original fraud rate: {self.y_train.mean():.4f}")
        print(f"✓ Resampled fraud rate: {self.y_train_resampled.mean():.4f}")
        
        # Convert back to DataFrame if needed
        if hasattr(self.X_train_resampled, 'shape') and len(self.X_train_resampled.shape) == 2:
            self.X_train_resampled = pd.DataFrame(
                self.X_train_resampled, 
                columns=self.X_train.columns
            )
    
    def train_random_forest(self):
        """Train Random Forest with improved parameters"""
        print("\n=== TRAINING RANDOM FOREST ===")
        
        # Use resampled data for training
        X_train_use = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_train_use = self.y_train_resampled if self.y_train_resampled is not None else self.y_train
        
        # Calculate class weights for the resampled data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train_use), 
            y=y_train_use
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Enhanced Random Forest parameters
        rf_model = RandomForestClassifier(
            n_estimators=300,           # More trees
            max_depth=20,               # Slightly deeper trees
            min_samples_split=10,       # Less restrictive
            min_samples_leaf=5,         # Less restrictive
            max_features='sqrt',        # Feature sampling
            class_weight='balanced',    # Let sklearn handle class weights
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True              # Out-of-bag scoring
        )
        
        # Train model
        print("Training Random Forest...")
        rf_model.fit(X_train_use, y_train_use)
        
        print(f"✓ Out-of-bag score: {rf_model.oob_score_:.4f}")
        
        # Cross-validation on original data
        cv_scores = cross_val_score(
            rf_model, X_train_use, y_train_use, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1'
        )
        
        print(f"✓ Cross-validation F1 scores: {cv_scores}")
        print(f"✓ Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        joblib.dump(rf_model, self.model_path / 'random_forest_model.pkl')
        self.models['random_forest'] = rf_model
        
        print("✓ Random Forest trained and saved")
        return rf_model
    
    def train_xgboost(self):
        """Train XGBoost with improved parameters"""
        print("\n=== TRAINING XGBOOST ===")
        
        # Use resampled data for training
        X_train_use = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_train_use = self.y_train_resampled if self.y_train_resampled is not None else self.y_train
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train_use[y_train_use == 0]) / len(y_train_use[y_train_use == 1])
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Enhanced XGBoost parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,           # More boosting rounds
            max_depth=8,                # Deeper trees
            learning_rate=0.05,         # Lower learning rate
            subsample=0.8,              # Subsample ratio
            colsample_bytree=0.8,       # Feature sampling
            scale_pos_weight=scale_pos_weight,  # Handle imbalanced data
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=0.1,             # L2 regularization
            min_child_weight=3          # Minimum child weight
        )
        
        # Train model
        print("Training XGBoost...")
        xgb_model.fit(X_train_use, y_train_use)
        
        # Cross-validation
        cv_scores = cross_val_score(
            xgb_model, X_train_use, y_train_use,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1'
        )
        
        print(f"✓ Cross-validation F1 scores: {cv_scores}")
        print(f"✓ Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        joblib.dump(xgb_model, self.model_path / 'xgboost_model.pkl')
        self.models['xgboost'] = xgb_model
        
        print("✓ XGBoost trained and saved")
        return xgb_model
    
    def evaluate_model(self, model, model_name):
        """Comprehensive model evaluation"""
        print(f"\n=== EVALUATING {model_name.upper()} ===")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Basic metrics
        try:
            report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Precision-Recall AUC (better for imbalanced data)
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Handle case where class 1 might not be predicted
            if '1' in report:
                precision_fraud = report['1']['precision']
                recall_fraud = report['1']['recall']
                f1_fraud = report['1']['f1-score']
            else:
                precision_fraud = 0.0
                recall_fraud = 0.0
                f1_fraud = 0.0
                print("Warning: No fraud cases predicted!")
            
            # Store results
            self.results[model_name] = {
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'precision': precision_fraud,
                'recall': recall_fraud,
                'f1_score': f1_fraud,
                'accuracy': report['accuracy']
            }
            
            # Print key metrics
            print(f"✓ Accuracy: {report['accuracy']:.4f}")
            print(f"✓ Precision: {precision_fraud:.4f}")
            print(f"✓ Recall: {recall_fraud:.4f}")
            print(f"✓ F1-Score: {f1_fraud:.4f}")
            print(f"✓ ROC AUC: {roc_auc:.4f}")
            print(f"✓ PR AUC: {pr_auc:.4f}")
            
            print("Confusion Matrix:")
            print(f"    Predicted:  0     1")
            print(f"Actual 0:     {cm[0][0]:4d}  {cm[0][1]:4d}")
            print(f"Actual 1:     {cm[1][0]:4d}  {cm[1][1]:4d}")
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            self.results[model_name] = {
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'pr_auc': 0.0
            }
        
        return self.results[model_name]
    
    def get_feature_importance(self, model, model_name):
        """Extract and save feature importance"""
        print(f"\nExtracting feature importance for {model_name}...")
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.X_train.columns
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'model': model_name
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            importance_path = self.results_path / f'{model_name}_feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            
            print(f"✓ Feature importance saved to {importance_path}")
            print(f"Top 10 most important features for {model_name}:")
            print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
            
            return importance_df
        else:
            print(f"Model {model_name} does not have feature importance")
            return None
    
    def compare_models(self):
        """Compare model performances"""
        print("\n=== MODEL COMPARISON ===")
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC AUC': results['roc_auc'],
                'PR AUC': results['pr_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison
        comparison_df.to_csv(self.results_path / 'model_comparison.csv', index=False)
        
        # Determine best model based on F1-score (good for fraud detection)
        best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_f1 = comparison_df['F1-Score'].max()
        
        print(f"\n✓ Best model: {best_model_name} (F1-Score: {best_f1:.4f})")
        
        return best_model_name, comparison_df
    
    def save_results(self):
        """Save all results to JSON"""
        results_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': {
                'train': self.X_train.shape,
                'test': self.X_test.shape
            },
            'fraud_rate': {
                'train': float(self.y_train.mean()),
                'test': float(self.y_test.mean())
            },
            'models': self.results
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for model_name in results_with_metadata['models']:
            if 'confusion_matrix' in results_with_metadata['models'][model_name]:
                cm = results_with_metadata['models'][model_name]['confusion_matrix']
                results_with_metadata['models'][model_name]['confusion_matrix'] = cm
        
        with open(self.results_path / 'training_results.json', 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"✓ Results saved to {self.results_path / 'training_results.json'}")
    
    def train_all_models(self):
        """Complete training pipeline with enhancements"""
        print("=== ENHANCED ENERGY FRAUD DETECTION - MODEL TRAINING ===\n")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Scale features
        self.scale_features()
        
        # Feature selection
        selected_features = self.feature_selection()
        
        # Apply resampling to handle class imbalance
        self.apply_resampling(strategy='smote')
        
        # Train models
        rf_model = self.train_random_forest()
        xgb_model = self.train_xgboost()
        
        # Evaluate models
        self.evaluate_model(rf_model, 'random_forest')
        self.evaluate_model(xgb_model, 'xgboost')
        
        # Get feature importance
        self.get_feature_importance(rf_model, 'random_forest')
        self.get_feature_importance(xgb_model, 'xgboost')
        
        # Compare models
        best_model, comparison = self.compare_models()
        
        # Save results
        self.save_results()
        
        print(f"\n✓ Model training completed successfully!")
        print(f"✓ Best performing model: {best_model}")
        print(f"✓ Models saved to: {self.model_path}")
        print(f"✓ Results saved to: {self.results_path}")
        
        return best_model, self.models, self.results

def main():
    """Main training function"""
    trainer = FraudModelTrainer()
    best_model, models, results = trainer.train_all_models()
    
    print("\n=== TRAINING SUMMARY ===")
    print(f"✓ Trained 2 models: Random Forest, XGBoost")
    print(f"✓ Best model: {best_model}")
    print(f"✓ Enhanced with: SMOTE resampling, feature selection, robust scaling")
    print(f"✓ Ready for Hour 4: Model evaluation and optimization")

if _name_ == "_main_":
    main()