import os

def create_lst_file(image_dir, lst_file_path):
    classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    class_to_label = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    with open(lst_file_path, 'w') as lst_file:
        index = 0
        for cls_name in classes:
            cls_path = os.path.join(image_dir, cls_name)
            for img_name in sorted(os.listdir(cls_path)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_name, img_name)
                    label = class_to_label[cls_name]
                    lst_file.write(f"{index}\t{label}\t{img_path}\n")
                    index += 1
    
    print(f"Đã tạo tệp .lst tại: {lst_file_path}")
    print(f"Tổng số hình ảnh: {index}")
    print(f"Số lớp: {len(classes)}")

if __name__ == "__main__":
    image_dir = '/home/asus/Downloads/SFace-main/eval/lfw/lfw_images/'  # Thư mục chứa các lớp
    lst_file_path = '/home/asus/FacialData/positive/train.lst'  # Đường dẫn tệp .lst muốn tạo
    create_lst_file(image_dir, lst_file_path)
