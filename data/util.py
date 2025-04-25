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

# cleanup_csv_files('/Users/aa/development/llm_test/a1/data/temp/combined_output.csv', '/Users/aa/development/llm_test/a1/data/temp/cleaned_combined_output.csv')

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

#example usage
# if __name__ == "__main__":
#     input_dir = '../data/temp/temp_row_*'
#     output_file = '../data/temp/combined_output.csv'
#     concatenate_csv_files(input_dir, output_file)


# Convert CSV to JSON
import csv
import json
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

# Example usage
# csv_file = 'data/temp/combined_output.csv'
# json_file = 'data/temp/combined_output.json'
# csv_to_json_pretty(csv_file, json_file)

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

# csv_file = 'data/temp/combined_output.csv'
# json_file = 'data/temp/combined_output.json'
# csv_to_json(csv_file, json_file)

def main():
    parser = argparse.ArgumentParser(description="Utility script for CSV operations.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for cleaning CSV files
    clean_parser = subparsers.add_parser("clean", help="Clean a CSV file")
    # clean_parser.add_argument("input", help="Path to the input CSV file")
    # clean_parser.add_argument("output", help="Path to the output cleaned CSV file")

    # Subparser for concatenating CSV files
    cat_parser = subparsers.add_parser("cat", help="Concatenate multiple CSV files")
    # cat_parser.add_argument("input_dir", help="Directory containing CSV files to concatenate")
    # cat_parser.add_argument("output", help="Path to the output concatenated CSV file")

    # Subparser for converting CSV to JSON
    to_json_parser = subparsers.add_parser("to_json", help="Convert a CSV file to JSON")
    # to_json_parser.add_argument("input", help="Path to the input CSV file")
    # to_json_parser.add_argument("output", help="Path to the output JSON file")
    # to_json_parser.add_argument("--delimiter", default=",", help="Delimiter used in the CSV file (default: ',')")

    args = parser.parse_args()

    if args.command == "clean":
        # cleanup_csv_files(args.input, args.output)
        # cleanup_csv_files('/Users/aa/development/llm_test/a1/data/temp/combined_output.csv', '/Users/aa/development/llm_test/a1/data/temp/cleaned_combined_output.csv')
        cleanup_csv_files('/Users/aa/development/llm_test/a1/data/train_ds_math_update.restore.csv', '/Users/aa/development/llm_test/a1/data/train_ds_math_update.restore.csv')
    elif args.command == "cat":
        # concatenate_csv_files(args.input_dir, args.output)
        input_dir = 'data/temp/temp_row_*'
        output_file = 'data/train_ds_math_update.restore.csv'
        concatenate_csv_files(input_dir, output_file)

    elif args.command == "to_json":
        # csv_to_json_pretty(args.input, args.output, args.delimiter)
        csv_file = 'data/train_ds_math_update.restore.csv'
        json_file = 'data/train_ds_math_update.restore.json'
        csv_to_json(csv_file, json_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()