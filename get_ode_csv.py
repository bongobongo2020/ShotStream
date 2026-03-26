import os
import csv
import argparse

def save_pt_paths_to_csv(root_folder, output_csv_path):
    # Check if input folder exists
    if not os.path.exists(root_folder):
        print(f"Error: Folder not found - {root_folder}")
        return

    print(f"Scanning directory: {root_folder}")
    
    count = 0
    # Open CSV file for writing
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write CSV header
        writer.writerow(['latent_path'])
        
        # Traverse directory recursively
        for root, dirs, files in os.walk(root_folder):
            for filename in files:
                # Filter .pt files
                if filename.endswith('.pt'):
                    # Get absolute path
                    absolute_path = os.path.abspath(os.path.join(root, filename))
                    writer.writerow([absolute_path])
                    count += 1
                    
    print(f"Done! Found {count} .pt files.")
    print(f"Saved to: {output_csv_path}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Find .pt files and save their paths to a CSV.")
    
    # Define input and output arguments
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to the input folder")
    parser.add_argument("-o", "--output_csv", type=str, required=True, help="Path to the output CSV file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run the function
    save_pt_paths_to_csv(args.input_dir, args.output_csv)