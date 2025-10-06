import json
import random
from PIL import Image
import numpy as np
random.seed(42)
np.random.seed(42)

def remove_words_by_ratio(text, ratio):
    words = text.split()
    num_to_remove = int(len(words) * ratio)
    indices_to_remove = set(random.sample(range(len(words)), num_to_remove))
    modified_words = [word for i, word in enumerate(words) if i not in indices_to_remove]
    return ' '.join(modified_words)

def generate_mae_mask(image_size, patch_size, mask_ratio):
    width, height = image_size
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size
    total_patches = num_patches_x * num_patches_y
    num_masked_patches = int(total_patches * mask_ratio)
    all_indices = np.arange(total_patches)
    np.random.shuffle(all_indices)
    masked_indices = all_indices[:num_masked_patches]
    return masked_indices.tolist()

def remove_http_from_pheme(text):
    if "http" in text:
        return text[:text.index("http")].strip()
    return text

for file in ["test.json", "train.json", "val.json"]:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for element in data:
            if 'text' in element:
                if 'http' in element['text']:
                    element['text'] = remove_http_from_pheme(element['text'])
                for mask_ratio in [0, 25, 50, 75, 100]:
                    ratio = 0.01 * mask_ratio
                    element["text_mask_" + str(mask_ratio)] = remove_words_by_ratio(element['text'], ratio)
            if 'image' in element:
                for mask_ratio in [0, 25, 50, 75, 100]:
                    ratio = 0.01 * mask_ratio
                    element["image_mask_" + str(mask_ratio)] = str(generate_mae_mask((224, 224), 32, ratio))

        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Processed JSON saved to {file}")
    except FileNotFoundError:
        print(f"Error: File {file} not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
