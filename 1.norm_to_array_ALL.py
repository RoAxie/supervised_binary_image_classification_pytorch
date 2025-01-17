# Convert normalized data files (.norm) into compressed NumPy array format (.npz) for efficient storage and faster access.

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, save_npz, load_npz
import tqdm


# path/
# ├── 0/
# │   ├── ATMO/
# │   └── PDK/
# ├── 1/
# │   ├── ATMO/
# │   └── PDK/
# └── 2/
#     ├── ATMO/
#     └── PDK/


# Create directories to store new .npz files: dir/plane/class
new_dir = 'new/rel/path'
os.makedirs(new_dir, exist_ok=True)

for i in range(3):
    plane_dir = os.path.join(new_dir, str(i))
    os.makedirs(plane_dir, exist_ok=True)
    os.makedirs(os.path.join(plane_dir, 'ATMO'), exist_ok=True)
    os.makedirs(os.path.join(plane_dir, 'PDK'), exist_ok=True)

print("Directory structure for storing .npz files successfully created.")


# Create a list of directories where normalized files are located
norm_dir = 'path/to/norm'
class_0_dirs = [f"{norm_dir}/{plane}/ATMO" for plane in range(3)]
class_1_dirs = [f"{norm_dir}/{plane}/PDK" for plane in range(3)]
class_dirs = [class_0_dirs, class_1_dirs]


# Convert each .norm file into .npz
for class_dir in class_dirs:
    atmo_pdk = class_dir[0].strip().split('/')[-1] # Only two values are allowed: ATMO or PDK
    for i in tqdm.tqdm(range(3)): # Iterate over directories for each plane
        for dir, subdirs, files in os.walk(class_dir[i]):
            if dir == class_dir[i]:
                for file in files: # Event projection from one (of three) readout plane
                    file_path = f"{class_dir[i]}/{file}"
                    base = os.path.splitext(file)[0] # Extract the file name without extension

                    with open(file_path, "r") as f:
                        image = lil_matrix((225, 225)) # Initialize a sparse matrix for the image
                        # Each line contains three values: position (hit wire), time, and voltage
                        for line in f:                   
                            fields = line.strip().split()
                            y, t, h = map(int, fields)
                            image[y, t] = h # Update the sparse matrix

                    new_file = f"{base}.npz"
                    save_npz(new_file, image.tocsr()) # Save the sparse matrix in compressed format

                    if 'plane0' in base:
                        destination_dir = f'new/rel/path/0/{atmo_pdk}'
                    elif 'plane1' in base:
                        destination_dir = f'new/rel/path/1/{atmo_pdk}'
                    else:
                        destination_dir = f'new/rel/path/2/{atmo_pdk}'
    
                    os.makedirs(destination_dir, exist_ok=True)
                    destination_path = f"{destination_dir}/{new_file}"
                    shutil.move(new_file, destination_path)

print('All .norm files have been successfully converted to .npz and saved in their respective directories.')



# Sample images

file_path = "path/to/npz_file"
sparse_matrix = load_npz(file_path)
dense_array = sparse_matrix.toarray()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax1.imshow(dense_array, cmap='gray', interpolation='none')
ax1.set_title('Grey scale')
ax1.axis('off')

im2 = ax2.imshow(dense_array, cmap='seismic', interpolation='none')
im2.set_clim(-np.max(np.abs(dense_array)), np.max(np.abs(dense_array))) 
ax2.set_title('Seismic')
ax2.axis('off')

fig.suptitle('Sample Images', fontsize=16)

output_image_path = "1b.sample_images.png"
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig(output_image_path, bbox_inches='tight')
print(f"Image saved as '{output_image_path}'")
plt.show()