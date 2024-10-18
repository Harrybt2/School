

import os

def convert_dat_file(file_path):
    # Create the new file name by changing the extension to .csv
    new_file_path = file_path.replace('.txt', '.csv')
    
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Replace the first line with 'x,y'
    lines[0] = "x,y\n"
    
    # Process the rest of the lines: delete first two spaces, replace space between numbers with a comma
    for i in range(1, len(lines)):
        # Strip the leading spaces and replace the space between numbers with a comma
        line = lines[i].strip()  # remove leading and trailing spaces
        numbers = line.split()  # split the numbers by space
        if len(numbers) == 2:  # if there are two numbers on the line
            lines[i] = f"{numbers[0]},{numbers[1]}\n"  # format as 'number1,number2'
    
    # Write the modified content to the new .csv file
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(lines)
    
    # Delete the original .dat file
    os.remove(file_path)

# Example usage
convert_dat_file('seligdatfile.txt')

