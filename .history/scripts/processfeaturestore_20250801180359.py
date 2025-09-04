import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    input_file = "/opt/ml/processing/input/Telco_customer_churn.xlsx"

    print(f"Reading Excel file from {input_file}...")
    df = pd.read_excel(input_file)

    # Optional: drop any unnecessary columns
    df.drop(columns=["CustomerID"], inplace=True, errors='ignore')

    # Drop rows with missing values just to be safe
    df = df.dropna()

    # Map Churn to binary if needed
    if "Churn Label" in df.columns:
        df["churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})
    elif "Churn Value" in df.columns:
        df["churn"] = df["Churn Value"]

    # Split
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    print("Saving to output directories...")

    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/validation", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)

    train.to_csv("/opt/ml/processing/train/train.csv", index=False)
    val.to_csv("/opt/ml/processing/validation/val.csv", index=False)
    test.to_csv("/opt/ml/processing/test/test.csv", index=False)

    print("✅ Done preprocessing and saving splits.")

if __name__ == "__main__":
    main()
