import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def detect_objects(image_path, conf_threshold=0.4):
    model = YOLO('yolov8n.pt')

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    res = model(image_rgb)[0]

    np.random.seed(57)
    colors = np.random.randint(0, 255, size=(len(res.names), 3), dtype=np.uint8)

    cls_names = res.names
    cls_lables = {}

    for box in res.boxes:
        if float(box.conf[0]) > conf_threshold:
            cls_id = int(box.cls[0])
            cls_name = cls_names[cls_id].title()
            color = colors[cls_id].tolist()
            pt1x, pt1y, pt2x, pt2y = map(int, box.xyxy[0])

            cv2.rectangle(
                img=image_rgb,
                pt1=(pt1x, pt1y),
                pt2=(pt2x, pt2y),
                color=color,
                thickness=4
            )

            if cls_lables.get(cls_name) is not None:
                cls_lables[cls_name][1] += 1
            else:
                cls_lables[cls_name] = [color, 1]

    return image_rgb, cls_lables


def main(image_path, conf_threshold=0.3, dpi=200):
    image, cls_lables = detect_objects(image_path, conf_threshold=conf_threshold)

    plt.figure(dpi=dpi)
    plt.imshow(image)
    plt.axis('off')

    legend_handles = []
    for cls_name, color_count in cls_lables.items():
        legend_handles.append(plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f'{cls_name} ({color_count[1]})',
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

    plt.savefig(f'detected_{image_path}', dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    main(image_path='PATH_TO_IMAGE')
