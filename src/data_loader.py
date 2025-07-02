import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
 
    def __init__(self, data_path='../data/'):
        self.data_path = Path(data_path)
        self.client_train = None
        self.invoice_train = None
        self.client_test = None
        self.invoice_test = None
        self.train_merged = None
    
    def load_all_data(self):
        """Load all CSV files"""
        print("Loading data files...")
        
        # Load with proper data types to avoid warnings
        self.client_train = pd.read_csv(self.data_path / 'client_train.csv')
        self.invoice_train = pd.read_csv(self.data_path / 'invoice_train.csv', low_memory=False)
        self.client_test = pd.read_csv(self.data_path / 'client_test.csv')
        self.invoice_test = pd.read_csv(self.data_path / 'invoice_test.csv', low_memory=False)
        
        print(f"✓ Loaded {len(self.client_train)} training customers")
        print(f"✓ Loaded {len(self.invoice_train)} training invoices")
        print(f"✓ Loaded {len(self.client_test)} test customers")
        print(f"✓ Loaded {len(self.invoice_test)} test invoices")
        
        return self.client_train, self.invoice_train, self.client_test, self.invoice_test
    
    def merge_training_data(self):
        if self.client_train is None or self.invoice_train is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        self.train_merged = self.invoice_train.merge(
            self.client_train, 
            on='client_id', 
            how='left'
        )
        
        print(f"✓ Merged dataset shape: {self.train_merged.shape}")
        
        # Validate merge
        missing_targets = self.train_merged['target'].isnull().sum()
        if missing_targets > 0:
            print(f"⚠️  Warning: {missing_targets} records without target labels")
        
        return self.train_merged
    
    def calculate_total_consumption(self):
        """Calculate total consumption from all tiers"""
        if self.train_merged is None:
            raise ValueError("Merged data not available. Call merge_training_data() first.")
        
        # Calculate total consumption across all tiers
        consumption_cols = ['consommation_level_1', 'consommation_level_2', 
                           'consommation_level_3', 'consommation_level_4']
        
        self.train_merged['total_consumption'] = self.train_merged[consumption_cols].sum(axis=1)
        
        print(f"✓ Added total_consumption column")
        return self.train_merged
    
    def get_basic_stats(self):
        """Get basic dataset statistics"""
        if self.client_train is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        stats = {
            'total_customers': len(self.client_train),
            'total_invoices': len(self.invoice_train),
            'fraud_rate': self.client_train['target'].mean(),
            'fraud_count': self.client_train['target'].sum(),
            'normal_count': (self.client_train['target'] == 0).sum(),
            'avg_invoices_per_customer': len(self.invoice_train) / len(self.client_train)
        }
        
        return stats
    
    def validate_data_quality(self):
        """Perform data quality checks"""
        if self.train_merged is None:
            raise ValueError("Merged data not available. Call merge_training_data() first.")
        
        quality_report = {}
        
        # Check for duplicates
        quality_report['duplicates'] = self.train_merged.duplicated().sum()
        
        # Check for missing values
        quality_report['missing_values'] = self.train_merged.isnull().sum().to_dict()
        
        # Check for negative consumption
        quality_report['negative_consumption'] = (
            self.train_merged['consommation_level_1'] < 0
        ).sum()
        
        # Check for zero consumption (total across all levels)
        if 'total_consumption' in self.train_merged.columns:
            quality_report['zero_consumption'] = (
                self.train_merged['total_consumption'] == 0
            ).sum()
        else:
            quality_report['zero_consumption'] = (
                self.train_merged['consommation_level_1'] == 0
            ).sum()
        
        # Check meter reading consistency
        quality_report['invalid_readings'] = (
            self.train_merged['new_index'] < self.train_merged['old_index']
        ).sum()
        
        return quality_report
    
    def get_fraud_insights(self):
        """Get fraud-specific insights"""
        if self.train_merged is None:
            raise ValueError("Merged data not available. Call merge_training_data() first.")
        
        fraud_data = self.train_merged[self.train_merged['target'] == 1]
        normal_data = self.train_merged[self.train_merged['target'] == 0]
        
        # Use total consumption if available, otherwise level 1
        consumption_col = 'total_consumption' if 'total_consumption' in self.train_merged.columns else 'consommation_level_1'
        
        insights = {
            'fraud_avg_consumption': fraud_data[consumption_col].mean(),
            'normal_avg_consumption': normal_data[consumption_col].mean(),
            'fraud_zero_consumption_rate': (fraud_data[consumption_col] == 0).mean(),
            'normal_zero_consumption_rate': (normal_data[consumption_col] == 0).mean(),
            'fraud_avg_invoices': fraud_data.groupby('client_id').size().mean(),
            'normal_avg_invoices': normal_data.groupby('client_id').size().mean(),
            'consumption_levels_analysis': {
                'fraud_level_1_avg': fraud_data['consommation_level_1'].mean(),
                'fraud_level_2_avg': fraud_data['consommation_level_2'].mean(),
                'fraud_level_3_avg': fraud_data['consommation_level_3'].mean(),
                'fraud_level_4_avg': fraud_data['consommation_level_4'].mean(),
                'normal_level_1_avg': normal_data['consommation_level_1'].mean(),
                'normal_level_2_avg': normal_data['consommation_level_2'].mean(),
                'normal_level_3_avg': normal_data['consommation_level_3'].mean(),
                'normal_level_4_avg': normal_data['consommation_level_4'].mean(),
            }
        }
        
        return insights
    
    def save_processed_data(self, output_path='../data/'):
        """Save processed data for next steps"""
        if self.train_merged is None:
            raise ValueError("No merged data to save. Call merge_training_data() first.")
        
        output_path = Path(output_path)
        self.train_merged.to_csv(output_path / 'train_merged.csv', index=False)
        print(f"✓ Saved merged training data to {output_path / 'train_merged.csv'}")

def main():
    print("=== ENERGY FRAUD DETECTION - DATA LOADING ===\n")
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load all data
    loader.load_all_data()
    
    # Merge training data
    loader.merge_training_data()
    
    # Calculate total consumption
    loader.calculate_total_consumption()
    
    # Get basic statistics
    stats = loader.get_basic_stats()
    print(f"\n=== BASIC STATISTICS ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    # Validate data quality
    quality = loader.validate_data_quality()
    print(f"\n=== DATA QUALITY REPORT ===")
    print(f"Duplicates: {quality['duplicates']}")
    print(f"Negative consumption: {quality['negative_consumption']}")
    print(f"Zero consumption: {quality['zero_consumption']}")
    print(f"Invalid readings: {quality['invalid_readings']}")
    
    # Get fraud insights
    insights = loader.get_fraud_insights()
    print(f"\n=== FRAUD INSIGHTS ===")
    print(f"Fraud avg consumption: {insights['fraud_avg_consumption']:.2f}")
    print(f"Normal avg consumption: {insights['normal_avg_consumption']:.2f}")
    print(f"Fraud zero consumption rate: {insights['fraud_zero_consumption_rate']:.2%}")
    print(f"Normal zero consumption rate: {insights['normal_zero_consumption_rate']:.2%}")
    
    # Save processed data
    loader.save_processed_data()
    

if __name__ == "__main__":
    main()