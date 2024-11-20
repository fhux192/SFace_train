import os
import pickle
import mxnet as mx

def create_lfw_bin(lfw_dir, pairs_path, output_path):
    bins = []
    issame_list = []

    # Read pairs.txt file
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first line
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                # Same person
                name = parts[0]
                idx1 = int(parts[1])
                idx2 = int(parts[2])
                img1_path = os.path.join(lfw_dir, f"{name}_{idx1:04d}.jpg")
                img2_path = os.path.join(lfw_dir, f"{name}_{idx2:04d}.jpg")
                issame = True
            elif len(parts) == 4:
                # Different people
                name1 = parts[0]
                idx1 = int(parts[1])
                name2 = parts[2]
                idx2 = int(parts[3])
                img1_path = os.path.join(lfw_dir, f"{name1}_{idx1:04d}.jpg")
                img2_path = os.path.join(lfw_dir, f"{name2}_{idx2:04d}.jpg")
                issame = False
            else:
                print(f"Invalid line format: {line}")
                continue

            # Check if image files exist
            if not os.path.exists(img1_path):
                print(f"Image not found: {img1_path}")
                continue
            if not os.path.exists(img2_path):
                print(f"Image not found: {img2_path}")
                continue

            # Read and process images as raw bytes
            try:
                with open(img1_path, 'rb') as img1_file:
                    img1_bytes = img1_file.read()
                with open(img2_path, 'rb') as img2_file:
                    img2_bytes = img2_file.read()
            except Exception as e:
                print(f"Error reading images: {e}")
                continue

            bins.append(img1_bytes)
            bins.append(img2_bytes)
            issame_list.append(issame)

    # Save to .bin file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"lfw.bin created at {output_path}")

if __name__ == "__main__":
    lfw_dir = '/home/asus/Downloads/SFace-main/eval/lfw/lfw-images'  # Update if different
    pairs_path = '/home/asus/Downloads/SFace-main/eval/lfw/pairs.txt'
    output_path = '/home/asus/Downloads/SFace-main/eval/lfw/lfw.bin'

    create_lfw_bin(lfw_dir, pairs_path, output_path)