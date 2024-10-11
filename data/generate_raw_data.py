import os
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random


def generate_random_transactions(
    users_df: pd.DataFrame, max_transactions: int = 11, max_days_back=365
) -> pd.DataFrame:
    # Predefined lists of categories and locations
    transaction_categories = [
        "Groceries",
        "Utilities",
        "Entertainment",
        "Dining",
        "Travel",
        "Health",
        "Education",
        "Shopping",
        "Automotive",
        "Rent",
    ]
    cities_and_states = [
        ("New York", "NY"),
        ("Los Angeles", "CA"),
        ("Chicago", "IL"),
        ("Houston", "TX"),
        ("Phoenix", "AZ"),
        ("Philadelphia", "PA"),
        ("San Antonio", "TX"),
        ("San Diego", "CA"),
        ("Dallas", "TX"),
        ("San Jose", "CA"),
    ]
    transactions_list = []

    for i, row in users_df.iterrows():
        num_transactions = np.random.randint(1, max_transactions)
        for j in range(num_transactions):
            # Random date within the last 10-max_days_back (default 365) days
            random_days = np.random.randint(10, max_days_back)
            date_of_transaction = datetime.now() - timedelta(days=random_days)
            city, state = random.choice(cities_and_states)
            if j == (num_transactions - 1):
                date_of_transaction == row["created"]

            transactions_list.append(
                {
                    "user_id": row["user_id"],
                    "created": date_of_transaction,
                    "updated": date_of_transaction,
                    "date_of_transaction": date_of_transaction,
                    "transaction_amount": round(np.random.uniform(10, 1000), 2),
                    "transaction_category": random.choice(transaction_categories),
                    "card_token": str(uuid.uuid4()),
                    "city": city,
                    "state": state,
                }
            )

    return pd.DataFrame(transactions_list)


def calculate_point_in_time_features(label_dataset, transactions_df) -> pd.DataFrame:
    label_dataset["created"] = pd.to_datetime(label_dataset["created"])
    transactions_df["transaction_timestamp"] = pd.to_datetime(
        transactions_df["date_of_transaction"]
    )

    # Get all transactions before the created time
    transactions_before = pd.merge(
        label_dataset[["user_id", "created"]], transactions_df, on="user_id"
    )
    transactions_before = transactions_before[
        transactions_before["transaction_timestamp"] < transactions_before["created_x"]
    ]
    transactions_before["days_between_transactions"] = (
        transactions_before["transaction_timestamp"] - transactions_before["created_x"]
    ).dt.days

    # Group by user_id and created to calculate features
    features = (
        transactions_before.groupby(["user_id", "created_x"])
        .agg(
            num_prev_transactions=("transaction_amount", "count"),
            avg_prev_transaction_amount=("transaction_amount", "mean"),
            max_prev_transaction_amount=("transaction_amount", "max"),
            stdv_prev_transaction_amount=("transaction_amount", "std"),
            days_since_last_transaction=("days_between_transactions", "min"),
            days_since_first_transaction=("days_between_transactions", "max"),
        )
        .reset_index()
        .fillna(0)
    )

    final_df = (
        pd.merge(
            label_dataset,
            features,
            left_on=["user_id", "created"],
            right_on=["user_id", "created_x"],
            how="left",
        )
        .reset_index(drop=True)
        .drop("created_x", axis=1)
    )

    return final_df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv(os.path.join(script_dir, "train.csv"))
    test = pd.read_csv(os.path.join(script_dir, "test.csv"))
    valid = pd.read_csv(os.path.join(script_dir, "validate.csv"))
    df = pd.concat([train, test, valid], axis=0).reset_index(drop=True)

    df["user_id"] = [f"user_{i}" for i in range(df.shape[0])]
    df["transaction_id"] = [f"txn_{i}" for i in range(df.shape[0])]

    for date_col in ["created", "updated"]:
        df[date_col] = pd.Timestamp.now()

    label_dataset = pd.DataFrame(
        df[
            [
                "user_id",
                "fraud",
                "created",
                "updated",
                "distance_from_home",
                "distance_from_last_transaction",
                "ratio_to_median_purchase_price",
            ]
        ]
    )

    user_purchase_history = generate_random_transactions(
        users_df=df[df["repeat_retailer"] == 1],
        max_transactions=5,
        max_days_back=365,
    )
    user_purchase_history.to_parquet("raw_transaction_datasource.parquet")
    finaldf = calculate_point_in_time_features(label_dataset, user_purchase_history)
    finaldf.to_parquet("final_data.parquet")


if __name__ == "__main__":
    main()
