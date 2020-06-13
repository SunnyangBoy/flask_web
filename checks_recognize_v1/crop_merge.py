import os
from PIL import Image

if __name__ == '__main__':
    mode1_dir = '/home/chen-ubuntu/Desktop/checks_dataset/crop_merge'
    mode2_dir = '/home/chen-ubuntu/Desktop/checks_dataset/stamp_merge2'
    #mode3_dir = '/home/chen-ubuntu/Desktop/checks_dataset/valid_crop_mode3'

    dirlist = [mode1_dir, mode2_dir]#, mode3_dir]

    merge_dir = '/home/chen-ubuntu/Desktop/checks_dataset/merge2'

    for i in range(1, 2):
        img_dir = os.path.join(dirlist[i], 'Image')
        img_files = os.listdir(img_dir)
        for img_file in sorted(img_files):
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path)
            new_path = os.path.join(merge_dir, 'Image', img_file)
            print(new_path)
            img.save(new_path)



