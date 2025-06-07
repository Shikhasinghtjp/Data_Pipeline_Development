
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Extracting the data
def extract_data(file_path='data.csv'):
    print("Extracting data...")
    df = pd.read_csv(file_path)
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    return df

#Transforming the data
def transform_data(df):
    print("Transforming data...")
    df = df.copy()

    # Replace '?' with NaN for numeric fields
    df.replace('?', pd.NA, inplace=True)

    # Clean and convert numeric columns
    df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('float')
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False)
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')  # convert to float, coercing non-numeric to NaN
    df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')

    # Parse 'Memory' column
    df['Memory'] = df['Memory'].str.replace('Flash Storage', 'SSD', regex=False)
    df['HDD'] = df['Memory'].apply(lambda x: 1 if 'HDD' in str(x) else 0)
    df['SSD'] = df['Memory'].apply(lambda x: 1 if 'SSD' in str(x) else 0)
    df.drop(columns=['Memory'], inplace=True)

    # Extract CPU brand
    df['Cpu_brand'] = df['Cpu'].apply(lambda x: str(x).split()[0] if pd.notna(x) else 'Unknown')
    df.drop(columns=['Cpu'], inplace=True)

    numeric_features = ['Inches', 'Ram', 'Weight']
    categorical_features = ['Company', 'TypeName', 'ScreenResolution', 'Gpu', 'OpSys', 'Cpu_brand']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X = df.drop('Price', axis=1)
    y = df['Price']

    X_transformed = preprocessor.fit_transform(X)

    transformed_df = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed,
        columns=preprocessor.get_feature_names_out()
    )

    final_df = pd.concat([transformed_df, df[['HDD', 'SSD']].reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return final_df

#Loading the data
def load_data(df, output_path='processed_data.csv'):
    print("Loading data...")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    df = extract_data('laptopData.csv')
    df_transformed = transform_data(df)
    load_data(df_transformed)
