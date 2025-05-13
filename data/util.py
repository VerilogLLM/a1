import os
import glob
import pandas as pd
import argparse
import csv
import json

def cleanup_csv_files(file_path, output_file_path):
    df = pd.read_csv(file_path)
    # Drop rows where both 'reasoning_plan' and 'reasoning_trace' columns are empty
    df_cleaned = df.dropna(subset=['reasoning_plan', 'reasoning_trace'], how='all')
    # Save the cleaned DataFrame back to the CSV file
    df_cleaned.to_csv(output_file_path, index=False)
    print("Rows with empty 'reasoning_plan' and 'reasoning_trace' columns have been removed.")

def cleanup_csv_file_leave_q_s(file_path, output_file_path):
    df = pd.read_csv(file_path)
    # Keep only the 'question' and 'solution' columns, set all other columns to NaN
    columns_to_keep = ['question', 'solution', 'data_source', 'category']
    for col in df.columns:
        if col not in columns_to_keep:
            df[col] = None

    # Save the modified DataFrame to the output file
    df.to_csv(output_file_path, index=False)
    print("All columns except 'question' and 'solution' have been emptied.")

def concatenate_csv_files(input_directory, output_file):
    # Use glob to find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
    
    # Initialize a list to hold the dataframes
    dataframes = []
    
    # Read each CSV file and append the dataframe to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Write the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)

def csv_to_json_pretty(csv_file_path, json_file_path, delimiter=','):
    # Initialize a list to hold the CSV rows
    data = []

    # Open and read the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
        
        # Iterate over each row in the CSV and add to the data list
        for row in csv_reader:
            data.append(row)

    # Write the data list to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Successfully converted {csv_file_path} to {json_file_path}")
    print(f"Number of rows in JSON: {len(data)}")

def csv_to_json(csv_file_path, json_file_path, delimiter=','):
    # Open the CSV file for reading
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
        
        # Open the JSON file for writing
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            # Iterate over each row in the CSV
            for row in csv_reader:
                # Write each row as a JSON object on a new line
                json_file.write(json.dumps(row) + '\n')

    print(f"Successfully converted {csv_file_path} to {json_file_path}")

def json_to_csv(json_file_path, csv_file_path):
    """
    Convert a newline-delimited JSON file to a CSV file.

    Args:
        json_file_path (str): Path to the input JSON file.
        csv_file_path (str): Path to the output CSV file.
    """
    # Open and read the JSON file
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        # Read each line as a separate JSON object
        data = [json.loads(line) for line in json_file]

    # Open the CSV file for writing
    with open(csv_file_path, mode='w', encoding='utf-8', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
        
        # Write the header row
        csv_writer.writeheader()
        
        # Write each row of data
        csv_writer.writerows(data)

    print(f"Successfully converted {json_file_path} to {csv_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Utility script for CSV operations.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for cleaning CSV files
    clean_parser = subparsers.add_parser("clean", help="Clean a CSV file")
    clean_parser = subparsers.add_parser("clean_leave_q_s", help="empty a CSV file except for question and solution")

    # Subparser for concatenating CSV files
    cat_parser = subparsers.add_parser("cat", help="Concatenate multiple CSV files")

    # Subparser for converting CSV to JSON
    to_json_parser = subparsers.add_parser("to_json", help="Convert a CSV file to JSON")
    to_csv_parser = subparsers.add_parser("to_csv", help="Convert a JSON file to csv")

    args = parser.parse_args()

    base_dir = os.getcwd()

    if args.command == "clean":
        cleanup_csv_files(
            os.path.join(base_dir, 'output/train_ds_math_update.restore.csv'),
            os.path.join(base_dir, 'output/train_ds_math_update.restore.csv')
        )
    elif args.command == "clean_leave_q_s":
        cleanup_csv_file_leave_q_s(
            os.path.join(base_dir, 'output/train_ds_math_update.csv'),
            os.path.join(base_dir, 'output/train_ds_math.csv')
        )
    elif args.command == "cat":
        input_dir = os.path.join(base_dir, 'output/temp/temp_row_*')
        output_file = os.path.join(base_dir, 'output/train_ds_math_update.restore.csv')
        concatenate_csv_files(input_dir, output_file)
    elif args.command == "to_json":
        # csv_file = os.path.join(base_dir, 'output/train_ds_math_update.restore.csv')
        # json_file = os.path.join(base_dir, 'output/train_ds_math_update.restore.json')
        csv_file = os.path.join(base_dir, 'output/train_ds_math.csv')
        json_file = os.path.join(base_dir, 'output/train_ds_math.json')
        csv_to_json(csv_file, json_file)
    elif args.command == "to_csv":
        # json_file = os.path.join(base_dir, 'output/train_ds_math.json')
        # csv_file = os.path.join(base_dir, 'output/train_ds_math.csv')
        json_file = os.path.join(base_dir, 'output/solution_non_matching_rows.json')
        csv_file = os.path.join(base_dir, 'output/solution_non_matching_rows.csv')
        json_to_csv(json_file, csv_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()