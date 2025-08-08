import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def detect_objects(image_paths, conf_threshold=0.4, objects=None):
    model = YOLO('yolov8n.pt')

    images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in image_paths]

    results = model(images)
    cls_names = results[0].names

    np.random.seed(57)
    colors = np.random.randint(0, 255, size=(len(cls_names), 3), dtype=np.uint8)

    for i in range(len(results)):
        cls_lables = {}
        for box in results[i].boxes:
            cls_id = int(box.cls[0])
            cls_name = cls_names[cls_id]

            if float(box.conf[0]) > conf_threshold and (not objects or cls_name in objects):
                color = colors[cls_id].tolist()
                pt1x, pt1y, pt2x, pt2y = map(int, box.xyxy[0])

                cv2.rectangle(
                    img=images[i],
                    pt1=(pt1x, pt1y),
                    pt2=(pt2x, pt2y),
                    color=color,
                    thickness=2
                )

                if cls_lables.get(cls_name) is not None:
                    cls_lables[cls_name][1] += 1
                else:
                    cls_lables[cls_name] = [color, 1]

        yield images[i], cls_lables


def main(image_paths, conf_threshold=0.4, dpi=200, objects=None):
    for i, (image, cls_lables) in enumerate(detect_objects(image_paths, conf_threshold, objects)):
        plt.figure(dpi=dpi)
        plt.imshow(image)
        plt.axis('off')

        legend_handles = []
        for cls_name, color_count in cls_lables.items():
            legend_handles.append(plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f'{cls_name.title()} ({color_count[1]})',
                markerfacecolor=np.array(color_count[0]) / 255.0,
                markersize=6,
                linestyle='None'
            ))

        plt.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            title='Objects',
            markerscale=1.5,
            handlelength=1.5,
            handleheight=1.2,
            labelspacing=0.7,
            borderpad=0.8,
        )

        plt.savefig(f'detected_{os.path.basename(image_paths[i])}', dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    filtered_objects = {'person', 'car'}
    batch = [
        'img.png',
        'img_1.png'
    ]
    main(image_paths=batch, conf_threshold=0.4, objects=filtered_objects)
