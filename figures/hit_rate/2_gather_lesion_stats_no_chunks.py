import os
import pandas as pd
import argparse
import re

def combine_csv_files(csv_directory, output_file):
    # List to store all the object data
    all_objects_data = []

    # Loop over all CSV files in the directory
    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(csv_directory, filename)

            # Read the CSV file
            df = pd.read_csv(file_path, header=None, skiprows=3)

            # Find the index where object data starts (assuming object data starts after a certain row)
            start_idx = df[df[0] == "Object number"].index[0] + 1

            # Extract object data (from the start index till the end)
            object_data = df.iloc[start_idx:].copy()

            # Add a column for the filename (optional, but can be useful to trace the origin)
            object_data['source_file'] = filename

            # Append to the list
            all_objects_data.append(object_data)

    # Combine all the objects data into one DataFrame
    combined_df = pd.concat(all_objects_data, ignore_index=True)

    # Save the combined data to a new CSV file
    combined_df.columns = ['Object number', 'Object volume', 'Detected flag', 'source_file']  # Setting proper column names
    combined_df.to_csv(output_file, index=False)

    print(f"Combined data saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Combine object data from multiple CSV files into a single CSV file.")
    
    # Add arguments for CSV directory and output file
    parser.add_argument('--csv_directory', type=str, required=True, help="Directory containing the CSV files.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output CSV file.")

    # Parse arguments
    args = parser.parse_args()

    # Run the combine function
    combine_csv_files(args.csv_directory, args.output_file)
