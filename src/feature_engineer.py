import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature Engineering for Energy Fraud Detection
    Creates customer-level features from invoice time series data
    """
    
    def __init__(self, data_path='../data/'):
        self.data_path = Path(data_path)
        self.merged_data = None
        self.customer_features = None
        
    def load_merged_data(self):
        """Load the preprocessed merged data"""
        print("Loading merged training data...")
        self.merged_data = pd.read_csv(self.data_path / 'train_merged.csv')
        print(f"✓ Loaded merged data: {self.merged_data.shape}")
        return self.merged_data
    
    def create_consumption_features(self):
        """Create consumption-based features for each customer"""
        print("Creating consumption features...")
        
        # Group by customer and calculate consumption statistics
        consumption_features = self.merged_data.groupby('client_id').agg({
            'total_consumption': [
                'mean',     # Average consumption
                'std',      # Consumption volatility (key fraud indicator)
                'min',      # Minimum consumption
                'max',      # Maximum consumption
                'count',    # Number of bills
                'sum'       # Total lifetime consumption
            ],
            'consommation_level_1': ['mean', 'std', 'sum'],
            'consommation_level_2': ['mean', 'std', 'sum'],
            'consommation_level_3': ['mean', 'std', 'sum'],
            'consommation_level_4': ['mean', 'std', 'sum']
        })
        
        # Flatten column names
        consumption_features.columns = [f"{col[0]}_{col[1]}" for col in consumption_features.columns]
        
        # Calculate additional consumption patterns
        consumption_features['consumption_volatility'] = (
            consumption_features['total_consumption_std'] / 
            (consumption_features['total_consumption_mean'] + 1)  # +1 to avoid division by zero
        )
        
        # Calculate consumption range (fraud indicator)
        consumption_features['consumption_range'] = (
            consumption_features['total_consumption_max'] - 
            consumption_features['total_consumption_min']
        )
        
        # Zero consumption frequency (strong fraud indicator)
        zero_consumption = self.merged_data.groupby('client_id').apply(
            lambda x: (x['total_consumption'] == 0).sum() / len(x)
        ).rename('zero_consumption_rate')
        
        consumption_features = consumption_features.join(zero_consumption)
        
        print(f"✓ Created {consumption_features.shape[1]} consumption features")
        return consumption_features
    
    def create_billing_features(self):
        """Create billing pattern features"""
        print("Creating billing pattern features...")
        
        billing_features = self.merged_data.groupby('client_id').agg({
            'months_number': [
                'mean',     # Average billing period
                'std',      # Billing period irregularity
                'min',      # Shortest billing period
                'max'       # Longest billing period
            ],
            'reading_remarque': [
                'nunique',  # Number of different reading remarks
                'count'     # Total number of readings
            ],
            'counter_statue': ['nunique'],  # Different meter statuses
            'counter_code': ['nunique'],    # Different counter codes
        })
        
        # Flatten column names
        billing_features.columns = [f"{col[0]}_{col[1]}" for col in billing_features.columns]
        
        # Calculate billing irregularity score
        billing_features['billing_irregularity'] = (
            billing_features['months_number_std'] / 
            (billing_features['months_number_mean'] + 1)
        )
        
        # Calculate reading remarks frequency (fraud indicator)
        billing_features['remarks_frequency'] = (
            billing_features['reading_remarque_nunique'] / 
            billing_features['reading_remarque_count']
        )
        
        print(f"✓ Created {billing_features.shape[1]} billing features")
        return billing_features
    
    def create_meter_features(self):
        """Create meter reading and index features"""
        print("Creating meter reading features...")
        
        # Calculate meter reading differences and patterns
        meter_data = self.merged_data.copy()
        meter_data['reading_difference'] = meter_data['new_index'] - meter_data['old_index']
        meter_data['coefficient_adjusted_consumption'] = (
            meter_data['total_consumption'] * meter_data['counter_coefficient']
        )
        
        meter_features = meter_data.groupby('client_id').agg({
            'reading_difference': ['mean', 'std', 'min', 'max'],
            'coefficient_adjusted_consumption': ['mean', 'std'],
            'counter_coefficient': ['mean', 'std', 'nunique'],
            'old_index': ['min', 'max'],
            'new_index': ['min', 'max']
        })
        
        # Flatten column names
        meter_features.columns = [f"{col[0]}_{col[1]}" for col in meter_features.columns]
        
        # Calculate meter progression (should be generally increasing)
        meter_features['meter_progression'] = (
            meter_features['new_index_max'] - meter_features['old_index_min']
        )
        
        # Detect potential meter rollbacks (fraud indicator)
        rollback_count = meter_data.groupby('client_id').apply(
            lambda x: (x['new_index'] < x['old_index']).sum()
        ).rename('meter_rollbacks')
        
        meter_features = meter_features.join(rollback_count)
        
        print(f"✓ Created {meter_features.shape[1]} meter features")
        return meter_features
    
    def create_temporal_features(self):
        """Create time-based features"""
        print("Creating temporal features...")
        
        # Convert invoice_date to datetime (handle different formats)
        temp_data = self.merged_data.copy()
        
        # Try to parse dates (they might be in different formats)
        try:
            temp_data['invoice_date'] = pd.to_datetime(temp_data['invoice_date'], errors='coerce')
        except:
            print("⚠️  Warning: Could not parse all dates")
            temp_data['invoice_date'] = pd.to_datetime(temp_data['invoice_date'], format='%d/%m/%Y', errors='coerce')
        
        # Drop rows with invalid dates
        temp_data = temp_data.dropna(subset=['invoice_date'])
        
        # Extract temporal components
        temp_data['year'] = temp_data['invoice_date'].dt.year
        temp_data['month'] = temp_data['invoice_date'].dt.month
        temp_data['quarter'] = temp_data['invoice_date'].dt.quarter
        
        temporal_features = temp_data.groupby('client_id').agg({
            'invoice_date': ['min', 'max', 'count'],
            'year': ['nunique'],
            'month': ['nunique'],
            'quarter': ['nunique']
        })
        
        # Flatten column names
        temporal_features.columns = [f"{col[0]}_{col[1]}" for col in temporal_features.columns]
        
        # Calculate customer tenure (days between first and last invoice)
        temporal_features['customer_tenure_days'] = (
            temporal_features['invoice_date_max'] - temporal_features['invoice_date_min']
        ).dt.days
        
        # Calculate billing frequency
        temporal_features['avg_days_between_bills'] = (
            temporal_features['customer_tenure_days'] / 
            (temporal_features['invoice_date_count'] - 1)
        ).fillna(0)
        
        print(f"✓ Created {temporal_features.shape[1]} temporal features")
        return temporal_features
    
    def create_customer_profile_features(self):
        """Create features from customer demographic data"""
        print("Creating customer profile features...")
        
        # Get unique customer information - check which columns exist first
        agg_dict = {}
        potential_cols = ['disrict', 'client_catg', 'region', 'creation_date', 'target']
        
        for col in potential_cols:
            if col in self.merged_data.columns:
                agg_dict[col] = 'first'
        
        customer_info = self.merged_data.groupby('client_id').agg(agg_dict)
        
        # Convert creation_date to datetime and extract features
        if 'creation_date' in customer_info.columns:
            try:
                customer_info['creation_date'] = pd.to_datetime(customer_info['creation_date'], format='%d/%m/%Y', errors='coerce')
                customer_info['account_age_years'] = (
                    datetime.now() - customer_info['creation_date']
                ).dt.days / 365.25
            except:
                print("⚠️  Warning: Could not parse creation dates")
                customer_info['account_age_years'] = 0
        else:
            customer_info['account_age_years'] = 0
        
        # Create categorical feature encodings
        categorical_cols = ['disrict', 'client_catg', 'region']
        available_cols = [col for col in categorical_cols if col in customer_info.columns]
        
        print(f"Customer info columns: {list(customer_info.columns)}")
        print(f"Available categorical columns: {available_cols}")
        
        # Initialize profile_features with the customer_info index
        profile_features = pd.DataFrame(index=customer_info.index)
        
        # Create dummy variables only if we have categorical columns
        if len(available_cols) > 0:
            print(f"Sample values: {customer_info[available_cols].head()}")
            
            # Debug: Check the actual data
            selected_data = customer_info[available_cols]
            print(f"Selected data shape: {selected_data.shape}")
            print(f"Selected data columns: {list(selected_data.columns)}")
            print(f"Any NaN values: {selected_data.isna().any().any()}")
            
            # Check if the selected DataFrame actually has columns
            if selected_data.shape[1] > 0:
                # Create dummy variables for each column individually to avoid prefix length issues
                dummy_dataframes = []
                
                for col in available_cols:
                    if col in selected_data.columns and not selected_data[col].isna().all():
                        # Create prefix for this column
                        if col == 'disrict':
                            prefix = 'district'
                        elif col == 'client_catg':
                            prefix = 'category'
                        elif col == 'region':
                            prefix = 'region'
                        else:
                            prefix = col
                        
                        print(f"Creating dummies for column: {col} with prefix: {prefix}")
                        col_dummies = pd.get_dummies(selected_data[col], prefix=prefix)
                        dummy_dataframes.append(col_dummies)
                
                # Combine all dummy DataFrames
                if dummy_dataframes:
                    dummy_features = pd.concat(dummy_dataframes, axis=1)
                    profile_features = pd.concat([profile_features, dummy_features], axis=1)
                    print(f"Created dummy features with shape: {dummy_features.shape}")
                else:
                    print("No valid categorical columns found for dummy creation")
            else:
                print("Selected data has no columns, skipping dummy creation")
        else:
            print("No categorical columns available")
        
        # Add account age
        profile_features['account_age_years'] = customer_info['account_age_years'].fillna(0)
        
        # Add target variable if it exists
        if 'target' in customer_info.columns:
            profile_features['target'] = customer_info['target']
        
        print(f"✓ Created {profile_features.shape[1]} customer profile features")
        return profile_features
    
    def create_fraud_specific_features(self):
        """Create features specifically designed to detect fraud patterns"""
        print("Creating fraud-specific features...")
        
        fraud_features = pd.DataFrame(index=self.merged_data['client_id'].unique())
        
        # 1. Sudden consumption drops (classic fraud pattern)
        consumption_by_customer = self.merged_data.groupby('client_id')['total_consumption'].apply(list)
        
        sudden_drops = []
        consumption_trends = []
        
        for client_id in fraud_features.index:
            consumptions = consumption_by_customer.get(client_id, [])
            if len(consumptions) > 2:
                # Calculate percentage drops between consecutive readings
                drops = []
                for i in range(1, len(consumptions)):
                    if consumptions[i-1] > 0:  # Avoid division by zero
                        drop_pct = (consumptions[i-1] - consumptions[i]) / consumptions[i-1]
                        drops.append(drop_pct)
                
                # Maximum drop percentage
                sudden_drops.append(max(drops) if drops else 0)
                
                # Overall trend (positive = increasing, negative = decreasing)
                if len(consumptions) > 1:
                    trend = (consumptions[-1] - consumptions[0]) / len(consumptions)
                    consumption_trends.append(trend)
                else:
                    consumption_trends.append(0)
            else:
                sudden_drops.append(0)
                consumption_trends.append(0)
        
        fraud_features['max_consumption_drop'] = sudden_drops
        fraud_features['consumption_trend'] = consumption_trends
        
        # 2. Billing pattern anomalies
        billing_gaps = self.merged_data.groupby('client_id')['months_number'].apply(
            lambda x: (x > x.median() * 2).sum()  # Count of unusually long billing periods
        )
        fraud_features['unusual_billing_gaps'] = billing_gaps
        
        # 3. Zero consumption streaks
        zero_streaks = self.merged_data.groupby('client_id').apply(
            lambda x: self._calculate_max_zero_streak(x['total_consumption'].values)
        )
        fraud_features['max_zero_consumption_streak'] = zero_streaks
        
        print(f"✓ Created {fraud_features.shape[1]} fraud-specific features")
        return fraud_features
    
    def _calculate_max_zero_streak(self, consumption_array):
        """Helper function to calculate maximum consecutive zeros"""
        if len(consumption_array) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for consumption in consumption_array:
            if consumption == 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def combine_all_features(self):
        """Combine all feature sets into final dataset"""
        print("\n=== COMBINING ALL FEATURES ===")
        
        # Create all feature sets
        consumption_features = self.create_consumption_features()
        billing_features = self.create_billing_features()
        meter_features = self.create_meter_features()
        temporal_features = self.create_temporal_features()
        profile_features = self.create_customer_profile_features()
        fraud_features = self.create_fraud_specific_features()
        
        # Combine all features
        print("Combining all feature sets...")
        
        # Start with consumption features as base
        combined_features = consumption_features.copy()
        
        # Join other feature sets
        feature_sets = [
            ('billing', billing_features),
            ('meter', meter_features),
            ('temporal', temporal_features),
            ('profile', profile_features),
            ('fraud', fraud_features)
        ]
        
        for name, features in feature_sets:
            print(f"Adding {name} features: {features.shape[1]} columns")
            combined_features = combined_features.join(features, how='left')
        
        # Handle missing values
        numeric_columns = combined_features.select_dtypes(include=[np.number]).columns
        combined_features[numeric_columns] = combined_features[numeric_columns].fillna(0)
        
        # Remove infinite values
        combined_features = combined_features.replace([np.inf, -np.inf], 0)
        
        self.customer_features = combined_features
        
        print(f"✓ Final feature set: {combined_features.shape}")
        print(f"✓ Features per customer: {combined_features.shape[1]}")
        print(f"✓ Customers: {combined_features.shape[0]}")
        
        return combined_features
    
    def get_feature_summary(self):
        """Get summary of created features"""
        if self.customer_features is None:
            raise ValueError("Features not created yet. Call combine_all_features() first.")
        
        summary = {
            'total_features': self.customer_features.shape[1],
            'total_customers': self.customer_features.shape[0],
            'fraud_cases': self.customer_features['target'].sum() if 'target' in self.customer_features.columns else 0,
            'feature_types': {
                'consumption': len([col for col in self.customer_features.columns if 'consumption' in col]),
                'billing': len([col for col in self.customer_features.columns if any(x in col for x in ['months', 'reading', 'counter'])]),
                'temporal': len([col for col in self.customer_features.columns if any(x in col for x in ['date', 'tenure', 'age'])]),
                'profile': len([col for col in self.customer_features.columns if any(x in col for x in ['district', 'category', 'region'])]),
                'fraud_specific': len([col for col in self.customer_features.columns if any(x in col for x in ['drop', 'streak', 'trend'])])
            }
        }
        
        return summary
    
    def save_features(self, output_path=None):
        """Save the engineered features"""
        if self.customer_features is None:
            raise ValueError("No features to save. Call combine_all_features() first.")
        
        if output_path is None:
            output_path = self.data_path / 'customer_features.csv'
        
        self.customer_features.to_csv(output_path)
        print(f"✓ Saved customer features to {output_path}")
        
        # Also save feature names for reference
        feature_names = pd.DataFrame({
            'feature_name': self.customer_features.columns,
            'feature_type': ['target' if col == 'target' else 'feature' for col in self.customer_features.columns]
        })
        
        feature_names_path = str(output_path).replace('.csv', '_metadata.csv')
        feature_names.to_csv(feature_names_path, index=False)
        print(f"✓ Saved feature metadata to {feature_names_path}")

def main():
    print("=== ENERGY FRAUD DETECTION - FEATURE ENGINEERING ===\n")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load merged data
    engineer.load_merged_data()
    
    # Create all features
    features = engineer.combine_all_features()
    
    # Get feature summary
    summary = engineer.get_feature_summary()
    print(f"\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Total features created: {summary['total_features']}")
    print(f"Total customers: {summary['total_customers']}")
    print(f"Fraud cases: {summary['fraud_cases']}")
    print(f"\nFeature breakdown:")
    for feature_type, count in summary['feature_types'].items():
        print(f"  {feature_type}: {count} features")
    
    # Show some key features
    print(f"\n=== KEY FEATURES SAMPLE ===")
    key_features = [
        'total_consumption_mean', 'consumption_volatility', 'zero_consumption_rate',
        'billing_irregularity', 'max_consumption_drop', 'max_zero_consumption_streak'
    ]
    
    available_key_features = [f for f in key_features if f in features.columns]
    if available_key_features:
        print(features[available_key_features + ['target']].head())
    
    # Save features
    engineer.save_features()
    
    print(f"\n✓ Feature engineering completed successfully!")
    print(f"✓ Ready for model training in Hour 3!")

if __name__ == "__main__":
    main()