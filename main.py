from src.data_validation import validate_dataset, plot_eda
from src.preprocessing import preprocess_data, balance_dataset
from src.model_training import train_models
from src.evaluation import evaluate_models
from src.metrics_visualization import visualize_metrics


def main():
    df = validate_dataset()
    plot_eda(df)
    df_clean = preprocess_data()
    # df_bal = balance_dataset(df_clean)  # opcional por RAM
    metrics = train_models()
    evaluate_models()
    visualize_metrics()


if __name__ == "__main__":
    main()
