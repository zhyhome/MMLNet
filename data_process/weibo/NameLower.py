import os

def rename_images_to_lowercase(folder_path):
    if not os.path.isdir(folder_path):
        print("指定的文件夹不存在")
        return

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            new_filename = filename.lower()  
            new_path = os.path.join(folder_path, new_filename)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")

folder_path = "/root/autodl-tmp/dataset/weibo/images"
rename_images_to_lowercase(folder_path)
