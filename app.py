import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder,
    PowerTransformer, KBinsDiscretizer, OrdinalEncoder, RobustScaler
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
import seaborn as sns
from datetime import datetime
from functools import wraps
import hashlib
import traceback
from flask_wtf.csrf import CSRFProtect
import json

# Initialize Flask app
app = Flask(__name__)
csrf = CSRFProtect(app)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
app.config.update(
    UPLOAD_FOLDER='uploads',
    PROCESSED_FOLDER='processed',
    ALLOWED_EXTENSIONS={'csv', 'xlsx', 'xls'},
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB
    SESSION_FILE_HASH='file_hash'
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_file_hash(filepath):
    """Generate MD5 hash of file contents for caching"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def validate_session_file(filename):
    """Check if file in session matches requested file"""
    return 'current_file' in session and session['current_file'] == filename

class DataProcessor:
    def __init__(self):
        self.history = []
        self.current_df = None
        self.original_df = None
        self.current_step = 1

    def to_dict(self):
        """Convert DataProcessor to a serializable dictionary"""
        return {
            'history': self.history,
            'current_df': self.current_df.to_dict(orient='records') if self.current_df is not None else None,
            'original_df': self.original_df.to_dict(orient='records') if self.original_df is not None else None,
            'current_step': self.current_step
        }

    @classmethod
    def from_dict(cls, data_dict):
        """Create DataProcessor from dictionary"""
        processor = cls()
        processor.history = data_dict.get('history', [])
        if data_dict.get('current_df'):
            processor.current_df = pd.DataFrame(data_dict['current_df'])
        if data_dict.get('original_df'):
            processor.original_df = pd.DataFrame(data_dict['original_df'])
        processor.current_step = data_dict.get('current_step', 1)
        return processor

    def add_history(self, step, action, details=""):
        """Record transformation history"""
        self.history.append({
            'step': step,
            'action': action,
            'details': details,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def load_data(self, filename):
        """Load data from file with validation"""
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format")
            
            self.original_df = df.copy()
            self.current_df = df
            self.add_history(0, "Data loaded", f"Shape: {df.shape}")
            return True
        except Exception as e:
            app.logger.error(f"Error loading file: {traceback.format_exc()}")
            return False

    def get_data_info(self):
        """Generate comprehensive data information"""
        if self.current_df is None:
            return None
            
        info = {
            'shape': self.current_df.shape,
            'dtypes': self.current_df.dtypes.astype(str).to_dict(),
            'null_counts': self.current_df.isnull().sum().to_dict(),
            'unique_counts': self.current_df.nunique().to_dict(),
            'numeric_cols': self.current_df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_cols': self.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        if len(info['numeric_cols']) > 1:
            info['correlation'] = self.current_df[info['numeric_cols']].corr().to_dict()
            
        return info

    def generate_visualizations(self):
        """Generate standard visualizations for the dataset"""
        if self.current_df is None:
            return None, None, None
            
        null_plot = dist_plot = corr_plot = None
        
        try:
            # Missing values heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(self.current_df.isnull(), cbar=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            null_plot = self._create_plot()
            
            # Distribution plots
            numeric_cols = self.current_df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                plt.figure(figsize=(12, 8))
                self.current_df[numeric_cols].hist(bins=20, layout=(-1, 3))
                plt.suptitle('Numeric Features Distribution')
                plt.tight_layout()
                dist_plot = self._create_plot()
                
                if len(numeric_cols) > 1:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(self.current_df[numeric_cols].corr(), annot=True, 
                                cmap='coolwarm', center=0, fmt=".2f")
                    plt.title('Correlation Heatmap')
                    corr_plot = self._create_plot()
                    
        except Exception as e:
            app.logger.error(f"Visualization error: {traceback.format_exc()}")
            
        return null_plot, dist_plot, corr_plot

    def _create_plot(self):
        """Convert matplotlib plot to base64 image"""
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url

    def clean_text_data(self, columns):
        """Clean text data in specified columns"""
        try:
            for col in columns:
                if col in self.current_df.columns and self.current_df[col].dtype == 'object':
                    self.current_df[col] = self.current_df[col].str.strip()
                    self.current_df[col] = self.current_df[col].str.replace(r'\s+', ' ', regex=True)
            
            self.add_history(2, "Text data cleaned", f"Columns: {', '.join(columns)}")
            return None
        except Exception as e:
            return f"Error cleaning text data: {str(e)}"

    def remove_duplicates(self):
        """Remove duplicate rows from dataset"""
        try:
            before = len(self.current_df)
            self.current_df = self.current_df.drop_duplicates()
            after = len(self.current_df)
            removed = before - after
            
            self.add_history(2, "Duplicates removed", f"Removed {removed} duplicate rows")
            return None
        except Exception as e:
            return f"Error removing duplicates: {str(e)}"

    def convert_data_types(self, type_mapping):
        """Convert data types according to mapping"""
        try:
            for col, new_type in type_mapping.items():
                if new_type == 'numeric':
                    self.current_df[col] = pd.to_numeric(self.current_df[col], errors='coerce')
                elif new_type == 'datetime':
                    self.current_df[col] = pd.to_datetime(self.current_df[col], errors='coerce')
                elif new_type == 'category':
                    self.current_df[col] = self.current_df[col].astype('category')
                elif new_type == 'string':
                    self.current_df[col] = self.current_df[col].astype(str)
            
            self.add_history(2, "Data types converted", f"Converted {len(type_mapping)} columns")
            return None
        except Exception as e:
            return f"Error converting data types: {str(e)}"

    def handle_missing_values(self, method, columns=None):
        """Handle missing values using specified method"""
        if columns is None:
            columns = [col for col in self.current_df.columns if self.current_df[col].isnull().sum() > 0]
        
        try:
            if method == 'mean':
                for col in columns:
                    self.current_df[col].fillna(self.current_df[col].mean(), inplace=True)
            elif method == 'median':
                for col in columns:
                    self.current_df[col].fillna(self.current_df[col].median(), inplace=True)
            elif method == 'mode':
                for col in columns:
                    self.current_df[col].fillna(self.current_df[col].mode()[0], inplace=True)
            elif method == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                self.current_df[columns] = imputer.fit_transform(self.current_df[columns])
            elif method == 'drop':
                self.current_df.dropna(subset=columns, inplace=True)
            elif method == 'ffill':
                self.current_df[columns] = self.current_df[columns].fillna(method='ffill')
            elif method == 'bfill':
                self.current_df[columns] = self.current_df[columns].fillna(method='bfill')
            elif method == 'interpolate':
                self.current_df[columns] = self.current_df[columns].interpolate()
            elif method == 'constant':
                for col in columns:
                    self.current_df[col].fillna('Unknown', inplace=True)
            
            self.add_history(3, "Missing values handled", f"Method: {method} | Columns: {columns}")
            return None
        except Exception as e:
            return f"Error handling missing values: {str(e)}"

    def transform_data(self, method, columns=None):
        """Transform data using specified method"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=['int64','float64']).columns.tolist()
        
        try:
            if method == 'standard':
                scaler = StandardScaler()
                self.current_df[columns] = scaler.fit_transform(self.current_df[columns])
            elif method == 'minmax':
                scaler = MinMaxScaler()
                self.current_df[columns] = scaler.fit_transform(self.current_df[columns])
            elif method == 'robust':
                scaler = RobustScaler()
                self.current_df[columns] = scaler.fit_transform(self.current_df[columns])
            elif method == 'log':
                for col in columns:
                    self.current_df[col] = np.log1p(self.current_df[col])
            elif method == 'boxcox':
                for col in columns:
                    self.current_df[col], _ = stats.boxcox(self.current_df[col]+1)
            
            self.add_history(4, "Data transformation applied", f"Method: {method} | Columns: {columns}")
            return None
        except Exception as e:
            return f"Error transforming data: {str(e)}"

    def encode_categorical(self, method, columns=None):
        """Encode categorical data"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        try:
            if method == 'label':
                encoder = LabelEncoder()
                for col in columns:
                    self.current_df[col] = encoder.fit_transform(self.current_df[col])
            elif method == 'onehot':
                self.current_df = pd.get_dummies(self.current_df, columns=columns)
            elif method == 'ordinal':
                encoder = OrdinalEncoder()
                self.current_df[columns] = encoder.fit_transform(self.current_df[columns])
            
            self.add_history(5, "Categorical encoding applied", f"Method: {method} | Columns: {columns}")
            return None
        except Exception as e:
            return f"Error encoding categorical data: {str(e)}"

    def handle_outliers(self, method, columns=None):
        """Handle outliers"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=['int64','float64']).columns.tolist()
        
        try:
            if method == 'remove':
                for col in columns:
                    q1 = self.current_df[col].quantile(0.25)
                    q3 = self.current_df[col].quantile(0.75)
                    iqr = q3 - q1
                    self.current_df = self.current_df[
                        (self.current_df[col] >= (q1 - 1.5*iqr)) & 
                        (self.current_df[col] <= (q3 + 1.5*iqr))
                    ]
            elif method == 'cap':
                for col in columns:
                    q1 = self.current_df[col].quantile(0.25)
                    q3 = self.current_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5*iqr
                    upper = q3 + 1.5*iqr
                    self.current_df[col] = self.current_df[col].clip(lower, upper)
            
            self.add_history(6, "Outliers handled", f"Method: {method} | Columns: {columns}")
            return None
        except Exception as e:
            return f"Error handling outliers: {str(e)}"

    def feature_engineering(self, methods, columns=None):
        """Feature engineering"""
        try:
            if 'datetime' in methods:
                datetime_cols = self.current_df.select_dtypes(include=['datetime']).columns
                for col in datetime_cols:
                    self.current_df[f'{col}_year'] = self.current_df[col].dt.year
                    self.current_df[f'{col}_month'] = self.current_df[col].dt.month
                    self.current_df[f'{col}_day'] = self.current_df[col].dt.day
            
            if 'binning' in methods and columns:
                binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                for col in columns:
                    self.current_df[f'{col}_binned'] = binner.fit_transform(self.current_df[[col]])
            
            self.add_history(7, "Feature engineering applied", f"Methods: {', '.join(methods)}")
            return None
        except Exception as e:
            return f"Error in feature engineering: {str(e)}"

    def dimensionality_reduction(self, method, n_components, target_col=None):
        """Dimensionality reduction"""
        try:
            numeric_cols = self.current_df.select_dtypes(include=['int64','float64']).columns
            
            if method == 'pca':
                pca = PCA(n_components=n_components)
                transformed = pca.fit_transform(self.current_df[numeric_cols])
                new_cols = [f'PC{i+1}' for i in range(n_components)]
                self.current_df = pd.concat([
                    self.current_df.drop(numeric_cols, axis=1),
                    pd.DataFrame(transformed, columns=new_cols)
                ], axis=1)
            elif method == 'selectkbest' and target_col:
                selector = SelectKBest(f_classif, k=n_components)
                selector.fit(self.current_df[numeric_cols], self.current_df[target_col])
                selected_cols = numeric_cols[selector.get_support()]
                self.current_df = self.current_df[selected_cols.tolist() + [target_col]]
            
            self.add_history(8, "Dimensionality reduction applied", f"Method: {method} | Components: {n_components}")
            return None
        except Exception as e:
            return f"Error in dimensionality reduction: {str(e)}"

    def final_type_conversion(self, type_mapping):
        """Final type conversion"""
        try:
            for col, new_type in type_mapping.items():
                if new_type == 'datetime':
                    self.current_df[col] = pd.to_datetime(self.current_df[col])
                elif new_type == 'numeric':
                    self.current_df[col] = pd.to_numeric(self.current_df[col])
                elif new_type == 'category':
                    self.current_df[col] = self.current_df[col].astype('category')
                elif new_type == 'boolean':
                    self.current_df[col] = self.current_df[col].astype(bool)
            
            self.add_history(9, "Final type conversion applied", f"Conversions: {len(type_mapping)} columns")
            return None
        except Exception as e:
            return f"Error in type conversion: {str(e)}"

    def validate_data(self):
        """Data validation"""
        validation_results = {}
        
        # Check for remaining nulls
        null_counts = self.current_df.isnull().sum()
        if null_counts.sum() > 0:
            validation_results['remaining_nulls'] = null_counts[null_counts > 0].to_dict()
        
        # Check numeric ranges
        numeric_cols = self.current_df.select_dtypes(include=['int64','float64']).columns
        for col in numeric_cols:
            if np.isinf(self.current_df[col]).any():
                validation_results[f'infinite_values_{col}'] = "Contains infinite values"
        
        self.add_history(10, "Data validation performed", f"Findings: {len(validation_results)} issues")
        return validation_results

    def export_data(self, format):
        """Export data"""
        try:
            processed_filename = f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            
            if format == 'csv':
                self.current_df.to_csv(processed_path, index=False)
            elif format == 'excel':
                self.current_df.to_excel(processed_path, index=False)
            
            self.add_history(11, "Data exported", f"Format: {format}")
            return processed_filename
        except Exception as e:
            return f"Error exporting data: {str(e)}"

# Route Helpers
def file_required(f):
    """Decorator to ensure valid file in session"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'current_file' not in session:
            flash('Please upload a file first', 'warning')
            return redirect(url_for('upload'))
        return f(*args, **kwargs)
    return decorated_function

def validate_step(step):
    """Validate that the current step matches the requested step"""
    if 'current_step' not in session or session['current_step'] != step:
        flash('Invalid step progression', 'danger')
        return False
    return True

class DataProcessorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DataProcessor):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = DataProcessorEncoder

# Routes
@app.route('/')
def index():
    """Home page with workflow overview"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file uploads with validation"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                session.clear()
                session['current_file'] = filename
                session['current_step'] = 1
                
                processor = DataProcessor()
                if not processor.load_data(filename):
                    flash('Error loading file. Please check the format.', 'danger')
                    return redirect(request.url)
                
                session['processor'] = processor.to_dict()
                return redirect(url_for('preview', filename=filename))
            except Exception as e:
                app.logger.error(f"Upload error: {traceback.format_exc()}")
                flash('Error uploading file', 'danger')
                
        flash('Invalid file type. Please upload CSV or Excel files.', 'danger')
        
    return render_template('upload.html')

@app.route('/preview/<filename>')
@file_required
def preview(filename):
    """Preview uploaded data with visualizations"""
    if not validate_session_file(filename):
        return redirect(url_for('upload'))
        
    if 'processor' not in session:
        flash('Session expired. Please upload your file again.', 'warning')
        return redirect(url_for('upload'))
        
    processor = DataProcessor.from_dict(session['processor'])
    data_info = processor.get_data_info()
    null_plot, dist_plot, corr_plot = processor.generate_visualizations()
    
    session['processor'] = processor.to_dict()
    
    return render_template('preprocessing.html',
                         filename=filename,
                         table=processor.current_df.head(20).to_html(classes='data-table', border=0),
                         data_info=data_info,
                         null_plot=null_plot,
                         dist_plot=dist_plot,
                         corr_plot=corr_plot,
                         current_step=session['current_step'],
                         total_steps=11,
                         transformation_history=processor.history)

@app.route('/process/<filename>/<int:step>', methods=['POST'])
@file_required
def process(filename, step):
    """Process data at each workflow step"""
    if not validate_step(step):
        return redirect(url_for('preview', filename=filename))
        
    if 'processor' not in session:
        flash('Session expired. Please upload your file again.', 'warning')
        return redirect(url_for('upload'))
        
    processor = DataProcessor.from_dict(session['processor'])
    process_details = {'action': 'No changes made'}
    
    try:
        if step == 1:  # Data Inspection
            processor.add_history(step, "Data inspection completed")
            
        elif step == 2:  # Data Cleaning
            cleaning_method = request.form.get('cleaning_method')
            if cleaning_method == 'text_clean':
                columns = request.form.getlist('cleaning_columns')
                error = processor.clean_text_data(columns)
                if error:
                    flash(error, 'danger')
            elif cleaning_method == 'remove_duplicates':
                processor.remove_duplicates()
            elif cleaning_method == 'convert_types':
                type_mapping = {}
                for col in processor.current_df.columns:
                    target_type = request.form.get(f'type_{col}')
                    if target_type and target_type != 'keep':
                        type_mapping[col] = target_type
                error = processor.convert_data_types(type_mapping)
                if error:
                    flash(error, 'danger')
        
        elif step == 3:  # Missing Values
            method = request.form.get('missing_method')
            columns = request.form.getlist('missing_columns')
            error = processor.handle_missing_values(method, columns or None)
            if error:
                flash(error, 'danger')
        
        elif step == 4:  # Data Transformation
            method = request.form.get('transform_method')
            columns = request.form.getlist('transform_columns')
            error = processor.transform_data(method, columns or None)
            if error:
                flash(error, 'danger')
        
        elif step == 5:  # Categorical Encoding
            method = request.form.get('encoding_method')
            columns = request.form.getlist('encoding_columns')
            error = processor.encode_categorical(method, columns or None)
            if error:
                flash(error, 'danger')
        
        elif step == 6:  # Outlier Handling
            method = request.form.get('outlier_method')
            columns = request.form.getlist('outlier_columns')
            error = processor.handle_outliers(method, columns or None)
            if error:
                flash(error, 'danger')
        
        elif step == 7:  # Feature Engineering
            methods = request.form.getlist('fe_methods')
            columns = request.form.getlist('fe_columns')
            error = processor.feature_engineering(methods, columns or None)
            if error:
                flash(error, 'danger')
        
        elif step == 8:  # Dimensionality Reduction
            method = request.form.get('dimred_method')
            n_components = int(request.form.get('n_components', 5))
            target_col = request.form.get('target_col')
            error = processor.dimensionality_reduction(method, n_components, target_col)
            if error:
                flash(error, 'danger')
        
        elif step == 9:  # Final Type Conversion
            type_mapping = {}
            for col in processor.current_df.columns:
                target_type = request.form.get(f'final_type_{col}')
                if target_type and target_type != 'keep':
                    type_mapping[col] = target_type
            error = processor.final_type_conversion(type_mapping)
            if error:
                flash(error, 'danger')
        
        elif step == 10:  # Data Validation
            validation_results = processor.validate_data()
            if validation_results:
                flash('Validation found issues in the data', 'warning')
        
        elif step == 11:  # Export Data
            format = request.form.get('export_format', 'csv')
            result = processor.export_data(format)
            if result.startswith('Error'):
                flash(result, 'danger')
            else:
                session['download_file'] = result
                return redirect(url_for('download', filename=result))
        
        # Update session and proceed to next step
        if step < 11:
            session['current_step'] = step + 1
        
        session['processor'] = processor.to_dict()
        return redirect(url_for('preview', filename=filename))
            
    except Exception as e:
        app.logger.error(f"Step {step} error: {traceback.format_exc()}")
        flash(f'Error processing step {step}: {str(e)}', 'danger')
        return redirect(url_for('preview', filename=filename))

@app.route('/download/<filename>')
@file_required
def download(filename):
    """Download processed data"""
    try:
        return send_from_directory(
            app.config['PROCESSED_FOLDER'],
            filename,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        app.logger.error(f"Download error: {traceback.format_exc()}")
        flash('Error downloading file', 'danger')
        return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the session and clear all data"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)