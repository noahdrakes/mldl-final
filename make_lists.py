import random


def train():
    # Define the input .list file containing the original file paths
    input_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/audio.list"

    # Define the directory to update the paths to
    new_directory = "/export/fs05/bodoom1/dl_proj/final_dl/list/xx/train"

    # Define the output .list file for the updated file paths
    output_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/audio.list"

    # Read the original file paths from the .list file
    with open(input_list_file, "r") as file:
        original_paths = file.readlines()

    # Process and update each file path
    updated_paths = []
    for path in original_paths:
        path = path.strip()  # Remove any leading/trailing whitespace or newlines
        if path:  # Ensure the path is not empty
            # Extract the filename from the original path and create a new path
            filename = path.split("/")[-1]
            updated_path = f"{new_directory}/{filename}"
            updated_paths.append(updated_path)

    # Write the updated paths to the output .list file
    with open(output_list_file, "w") as file:
        file.write("\n".join(updated_paths))

    print(f"Updated paths have been written to {output_list_file}")

    # Define the input .list file containing the original file paths
    input_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/rgb.list"

    # Define the directory to update the paths to
    new_directory = "/export/fs05/bodoom1/dl_proj/final_dl/dl_files/i3d-features/RGB"

    # Define the output .list file for the updated file paths
    output_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/rgb.list"

    # Read the original file paths from the .list file
    with open(input_list_file, "r") as file:
        original_paths = file.readlines()

    # Process and update each file path
    updated_paths = []
    for path in original_paths:
        path = path.strip()  # Remove any leading/trailing whitespace or newlines
        if path:  # Ensure the path is not empty
            # Extract the filename from the original path and create a new path
            filename = path.split("/")[-1]
            updated_path = f"{new_directory}/{filename}"
            updated_paths.append(updated_path)

    # Write the updated paths to the output .list file
    with open(output_list_file, "w") as file:
        file.write("\n".join(updated_paths))

    print(f"Updated paths have been written to {output_list_file}")


def test():
    # Define the input .list file containing the original file paths
    input_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/audio_test.list"

    # Define the directory to update the paths to
    new_directory = "/export/fs05/bodoom1/dl_proj/final_dl/list/xx/test"

    # Define the output .list file for the updated file paths
    output_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/audio_test.list"

    # Read the original file paths from the .list file
    with open(input_list_file, "r") as file:
        original_paths = file.readlines()

    # Process and update each file path
    updated_paths = []
    for path in original_paths:
        path = path.strip()  # Remove any leading/trailing whitespace or newlines
        if path:  # Ensure the path is not empty
            # Extract the filename from the original path and create a new path
            filename = path.split("/")[-1]
            updated_path = f"{new_directory}/{filename}"
            updated_paths.append(updated_path)

    # Write the updated paths to the output .list file
    with open(output_list_file, "w") as file:
        file.write("\n".join(updated_paths))

    print(f"Updated paths have been written to {output_list_file}")

    # Define the input .list file containing the original file paths
    input_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/rgb_test.list"

    # Define the directory to update the paths to
    new_directory = "/export/fs05/bodoom1/dl_proj/final_dl/dl_files/i3d-features/RGBTest"

    # Define the output .list file for the updated file paths
    output_list_file = "/export/fs05/bodoom1/dl_proj/final_dl/list/rgb_test.list"

    # Read the original file paths from the .list file
    with open(input_list_file, "r") as file:
        original_paths = file.readlines()

    # Process and update each file path
    updated_paths = []
    for path in original_paths:
        path = path.strip()  # Remove any leading/trailing whitespace or newlines
        if path:  # Ensure the path is not empty
            # Extract the filename from the original path and create a new path
            filename = path.split("/")[-1]
            updated_path = f"{new_directory}/{filename}"
            updated_paths.append(updated_path)

    # Write the updated paths to the output .list file
    with open(output_list_file, "w") as file:
        file.write("\n".join(updated_paths))

    print(f"Updated paths have been written to {output_list_file}")




import random

def split_file(input_file, output_file_80, output_file_20):
    # Read all lines from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Shuffle the lines to randomize them
    random.shuffle(lines)
    
    # Calculate split index
    split_index = int(len(lines) * 0.8)
    
    # Split into 80% and 20%
    lines_80 = lines[:split_index]
    lines_20 = lines[split_index:]
    
    # Write the 80% lines to the output file for 80%
    with open(output_file_80, 'w') as file:
        file.writelines(lines_80)
    
    # Write the 20% lines to the output file for 20%
    with open(output_file_20, 'w') as file:
        file.writelines(lines_20)
    
    print(f"Split complete. {len(lines_80)} lines written to {output_file_80}.")
    print(f"{len(lines_20)} lines written to {output_file_20}.")

train()
test()
split_file("/export/fs05/bodoom1/dl_proj/final_dl/list/audio.list", "/export/fs05/bodoom1/dl_proj/final_dl/list/audio_train.list", "/export/fs05/bodoom1/dl_proj/final_dl/list/audio_val.list")
split_file("/export/fs05/bodoom1/dl_proj/final_dl/list/rgb.list", "/export/fs05/bodoom1/dl_proj/final_dl/list/rgb_train.list", "/export/fs05/bodoom1/dl_proj/final_dl/list/rgb_val.list")