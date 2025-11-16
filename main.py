import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DementiaRiskPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load the dataset and explore its structure"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            
            # Display basic info about the dataset
            print("\nDataset Info:")
            print(f"Number of columns: {len(self.df.columns)}")
            print(f"Number of rows: {len(self.df)}")
            
            return True
        except FileNotFoundError:
            print("Data file not found. Please check the file path.")
            return False
    
    def explore_target_variable(self, target_column='NACCUDSD'):
        """Explore the target variable distribution"""
        if target_column in self.df.columns:
            print(f"\nTarget variable '{target_column}' distribution:")
            target_counts = self.df[target_column].value_counts().sort_index()
            print(target_counts)
            
            # Show what each value means (you'll need to check your data dictionary)
            print("\nTarget value meanings (check your data dictionary):")
            print("1 = Normal")
            print("2 = MCI (Mild Cognitive Impairment)") 
            print("3 = Dementia")
            print("4 = Impaired, not MCI")
            
            return target_counts
        else:
            print(f"Target column '{target_column}' not found.")
            return None
    
    def create_binary_target(self, target_column='NACCUDSD'):
        """Create binary target: 0 = No Dementia, 1 = Dementia"""
        # Based on NACC UDS coding: 1=Normal, 2=MCI, 3=Dementia, 4=Impaired not MCI
        self.df['dementia_binary'] = (self.df[target_column] == 3).astype(int)
        
        print("Binary target variable created:")
        print(f"Non-Dementia (0): {(self.df['dementia_binary'] == 0).sum()}")
        print(f"Dementia (1): {(self.df['dementia_binary'] == 1).sum()}")
        
        return 'dementia_binary'
    
    def identify_features(self):
        """Identify relevant features from the available columns"""
        # Based on the columns we found
        feature_mapping = {
            'age': 'NACCAGE',
            'education': 'EDUC',
            'gender': 'SEX',  # Using SEX instead of NACCGEND
            'stroke': 'STROKE',
            'heart_attack': 'CVHATT',
            'alcohol': 'ALCOHOL',
            'bmi': 'NACCBMI',  # This might be calculated BMI
            'height': 'HEIGHT',
            'weight': 'WEIGHT',
            'hypertension': 'HYPERT',
            'diabetes': 'DIABET',
            'smoking': 'TOBAC100',  # Ever smoked 100 cigarettes?
            'depression': 'DEP2YRS'
        }
        
        # Only keep features that actually exist in our dataset
        available_features = {}
        for feature_type, col_name in feature_mapping.items():
            if col_name in self.df.columns:
                available_features[feature_type] = col_name
                print(f"✅ {feature_type.upper()}: {col_name}")
            else:
                print(f"❌ {feature_type.upper()}: {col_name} not found")
        
        return available_features
    
    def preprocess_data(self, feature_mapping, target_column='dementia_binary'):
        """Preprocess the data with available features"""
        df_processed = self.df.copy()
        
        print("\nPreprocessing steps:")
        
        # Handle missing values for each feature
        for feature_type, col_name in feature_mapping.items():
            if col_name in df_processed.columns:
                if df_processed[col_name].dtype in ['int64', 'float64']:
                    # Numerical column - fill with median
                    median_val = df_processed[col_name].median()
                    df_processed[col_name].fillna(median_val, inplace=True)
                    print(f"  - Filled missing values in {col_name} with median: {median_val}")
                else:
                    # Categorical column - fill with mode
                    mode_val = df_processed[col_name].mode()[0] if not df_processed[col_name].mode().empty else 'Unknown'
                    df_processed[col_name].fillna(mode_val, inplace=True)
                    print(f"  - Filled missing values in {col_name} with mode: {mode_val}")
        
        # Feature Engineering
        # Create Health Risk Index from available risk factors
        risk_factors = []
        if 'stroke' in feature_mapping:
            risk_factors.append(feature_mapping['stroke'])
        if 'heart_attack' in feature_mapping:
            risk_factors.append(feature_mapping['heart_attack'])
        if 'hypertension' in feature_mapping:
            risk_factors.append(feature_mapping['hypertension'])
        if 'diabetes' in feature_mapping:
            risk_factors.append(feature_mapping['diabetes'])
        
        if risk_factors:
            df_processed['Health_Risk_Index'] = 0
            for risk_factor in risk_factors:
                if df_processed[risk_factor].dtype == 'object':
                    # Convert yes/no to 1/0
                    df_processed['Health_Risk_Index'] += (
                        df_processed[risk_factor].astype(str).str.upper().isin(['YES', '1', 'TRUE', 'CURRENT']).astype(int)
                    )
                else:
                    # Assume 1 indicates presence of risk factor
                    df_processed['Health_Risk_Index'] += (df_processed[risk_factor] == 1).astype(int)
            print(f"  - Created Health_Risk_Index from {len(risk_factors)} risk factors")
        
        # Create Age Groups
        if 'age' in feature_mapping:
            age_col = feature_mapping['age']
            df_processed['Age_Group'] = pd.cut(df_processed[age_col], 
                                             bins=[0, 59, 69, 79, 100], 
                                             labels=[0, 1, 2, 3])  # 0=<60, 1=60-69, 2=70-79, 3=80+
            print("  - Created Age_Group feature")
        
        self.df_processed = df_processed
        self.feature_mapping = feature_mapping
        self.target_column = target_column
        
        print(f"Preprocessing completed. Processed dataframe shape: {df_processed.shape}")
        
        return df_processed
    
    def prepare_features(self, feature_list=None, target_column='dementia_binary'):
        """Prepare features for training"""
        if feature_list is None:
            # Create default feature set based on available features
            feature_list = []
            
            # Add core features
            core_features = ['age', 'education', 'gender', 'Health_Risk_Index', 'Age_Group']
            for feature in core_features:
                if feature in self.feature_mapping:
                    col_name = self.feature_mapping[feature] if feature in ['age', 'education', 'gender'] else feature
                    if col_name in self.df_processed.columns:
                        feature_list.append(col_name)
            
            # Add any additional available features
            additional_features = ['stroke', 'heart_attack', 'hypertension', 'diabetes', 'bmi']
            for feature in additional_features:
                if feature in self.feature_mapping and self.feature_mapping[feature] not in feature_list:
                    feature_list.append(self.feature_mapping[feature])
        
        print(f"Using features: {feature_list}")
        print(f"Using target: {target_column}")
        
        X = self.df_processed[feature_list]
        y = self.df_processed[target_column]
        
        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
            self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        print(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models"""
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                print(f"✅ {name} trained successfully")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        print("Model training completed.")
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models with binary classification metrics"""
        if not self.models:
            print("No models trained yet. Please run train_models() first.")
            return None
            
        results = []
        
        for name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate binary classification metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='binary', zero_division=0)
                
                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy, 4),
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1-Score': round(f1, 4)
                })
                
                print(f"\n{name} Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                print("\nClassification Report:")
                print(classification_report(self.y_test, y_pred, target_names=['No Dementia', 'Dementia']))
                
                # Confusion Matrix
                cm = confusion_matrix(self.y_test, y_pred)
                print("Confusion Matrix:")
                print(cm)
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return pd.DataFrame(results)
    
    def predict_risk(self, model_name, features):
        """Predict dementia risk for new data"""
        if model_name in self.models:
            model = self.models[model_name]
            # Preprocess features similarly to training
            features_processed = self.scaler.transform([features])
            probability = model.predict_proba(features_processed)[0, 1]
            
            risk_level = "High Risk" if probability > 0.5 else "Low Risk"
            
            return {
                'probability': probability,
                'risk_level': risk_level,
                'percentage': f"{probability * 100:.1f}%"
            }
        else:
            print(f"Model {model_name} not found.")
            return None

# Main execution
if __name__ == "__main__":
    # Initialize the predictor
    predictor = DementiaRiskPredictor()
    
    # Load your data
    if predictor.load_data('data/raw data/.gitkeep data'):  # Update this path if needed
        # Explore the target variable
        print("\n" + "="*50)
        print("STEP 1: Exploring target variable...")
        print("="*50)
        target_distribution = predictor.explore_target_variable('NACCUDSD')
        
        # Create binary target variable
        print("\n" + "="*50)
        print("STEP 2: Creating binary target variable...")
        print("="*50)
        binary_target = predictor.create_binary_target('NACCUDSD')
        
        # Identify available features
        print("\n" + "="*50)
        print("STEP 3: Identifying available features...")
        print("="*50)
        feature_mapping = predictor.identify_features()
        
        # Preprocess data
        print("\n" + "="*50)
        print("STEP 4: Preprocessing data...")
        print("="*50)
        predictor.preprocess_data(feature_mapping, binary_target)
        
        # Prepare features
        print("\n" + "="*50)
        print("STEP 5: Preparing features...")
        print("="*50)
        predictor.prepare_features()
        
        # Train models
        print("\n" + "="*50)
        print("STEP 6: Training models...")
        print("="*50)
        predictor.train_models()
        
        # Evaluate models
        print("\n" + "="*50)
        print("STEP 7: Evaluating models...")
        print("="*50)
        results_df = predictor.evaluate_models()
        
        if results_df is not None:
            print("\n" + "="*50)
            print("FINAL MODEL COMPARISON")
            print("="*50)
            print(results_df.to_string(index=False))
            
            # Show feature importance for Random Forest
            if 'Random Forest' in predictor.models:
                print("\n" + "="*50)
                print("FEATURE IMPORTANCE (Random Forest)")
                print("="*50)
                rf_model = predictor.models['Random Forest']
                feature_importance = pd.DataFrame({
                    'feature': predictor.X_train.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(feature_importance.head(10))