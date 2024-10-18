# for lab 4 of measuremnts 2/15/2024
import csv

def import_data(file_path):
    positions = []
    accelerations = {'x': [], 'y': [], 'z': []}
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')  # assuming tab-separated values
        for row in reader:
            try:
                position = float(row[0])
                accel_x = float(row[1])
                accel_y = float(row[2])
                accel_z = float(row[3])
                
                positions.append(position)
                accelerations['x'].append(accel_x)
                accelerations['y'].append(accel_y)
                accelerations['z'].append(accel_z)
                
            except (ValueError, IndexError):
                print("Invalid data format or missing values. Skipping row.")
    
    return positions, accelerations

if __name__ == "__main__":
    file_path = "D:\Documents\Measurements\Acceleration without g 2024-02-15 12-17-35\Raw Data.csv"  # Change this to the path of your CSV file
    position_data, acceleration_data = import_data(file_path)
    
    print("Positions:", position_data)
    print("Accelerations:")
    print("  x:", acceleration_data['x'])
    print("  y:", acceleration_data['y'])
    print("  z:", acceleration_data['z'])
