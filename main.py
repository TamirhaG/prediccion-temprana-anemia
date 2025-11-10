from src.data_validation import validate_dataset, plot_eda
from src.preprocessing import preprocess_data, balance_dataset
from src.model_training import train_models

def main():
    df = validate_dataset()
    plot_eda(df)
    df_clean = preprocess_data()
    df_bal = balance_dataset(df_clean)
    metrics = train_models()

if __name__ == "__main__":
    main()
