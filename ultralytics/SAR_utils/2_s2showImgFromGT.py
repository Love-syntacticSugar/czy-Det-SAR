import os
import cv2

# SEEME 一般来说用不到这和脚本，因为v8里面有
def read_annotations(filename):
    annotations = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_name = parts[0]
            boxes = list(map(int, parts[1:]))
            if image_name in annotations:
                annotations[image_name].append(boxes)
            else:
                annotations[image_name] = [boxes]
    return annotations

def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), 2)
        cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.line(image, (x4, y4), (x1, y1), (0, 255, 0), 2)

def main():
    annotations = read_annotations('groundtruth.txt')
    image_folder = 'JPEGImages'
    output_folder = 's2mask'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name, boxes in annotations.items():
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        draw_boxes(image, boxes)
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image)

if __name__ == '__main__':
    main()