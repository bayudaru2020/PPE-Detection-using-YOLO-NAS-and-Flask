import argparse
import os
import cv2
from flask import Flask, render_template, request, send_from_directory, jsonify
import numpy as np
from super_gradients.training import models
import torch
from flask import send_file

import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes

app = Flask(__name__, static_folder='static')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

dataset_params = {
    'classes': ['Mask', 'Wear', 'Glove', 'Boots', 'Helmet', 'Vest', 'Shield']
}
model = models.get(
    'yolo_nas_s',
    num_classes=len(dataset_params['classes']),
    #pretrained_weights="coco"
    checkpoint_path="model/yolo-nas-finetuned.pth"
).to(device)

label_color_map = {
    'Mask' : ((255, 0, 0), (255, 255, 255)), #merah-putih
    'Wear' : ((0, 255, 0), (0, 0, 0)), #hijau-hitam
    'Glove': ((0, 0, 255), (255, 255, 255)), #biru-putih
    'Boots': ((255, 255, 0), (0, 0, 0)), #kuning-hitam
    'Helmet': ((255, 0, 255), (255, 255, 255)), #magenta-putih
    'Vest': ((0, 255, 255), (0, 0, 0)), #cyan-hitam
    'Shield': ((128, 128, 128), (255, 255, 255)) #abu-putih
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filepath = os.path.join('uploads', f.filename)
            f.save(filepath)
            file_extension = f.filename.rsplit('.', 1)[1].lower()


            if file_extension == 'jpg':
                gambar = cv2.imread(filepath)
                detections = model.predict(gambar)

            #mengubah hasil deteksi model ke bentuk yg dapat disimpan
                if detections:
                    img = detections.image  #Mengambil gambar dari objek detections
                    if detections.prediction.bboxes_xyxy is not None:
                        for i, bbox in enumerate(detections.prediction.bboxes_xyxy):
                            label = dataset_params['classes'][int(detections.prediction.labels[i])]
                            confidence = float(detections.prediction.confidence[i])

                            #mengubah gambar dalam format yang bisa diolah oleh OpenCV
                            img = np.array(img, dtype=np.uint8)

                            label_color, text_color = label_color_map.get(label, ((0, 0, 0), (255, 255, 255)))

                            #mengubah format bounding box menjadi sesuai dengan cv2.rectangle
                            bbox_pt1 = (int(bbox[0]), int(bbox[1]))
                            bbox_pt2 = (int(bbox[2]), int(bbox[3]))

                            cv2.rectangle(img, bbox_pt1, bbox_pt2, label_color, 1)
                            cv2.rectangle(img, (bbox_pt1[0], bbox_pt1[1] - 20), (bbox_pt1[0] + 120, bbox_pt1[1]), label_color, -1)#color bacground text/dibuat rextangle
                            cv2.putText(img, f"{label} {confidence:.2f}", (bbox_pt1[0], bbox_pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                #menyimpan img
                annotated_image_path = os.path.join('static','runs', 'detect', 'jpg', 'result.jpg')
                cv2.imwrite(annotated_image_path, img)

                return send_file('static/runs/detect/jpg/result.jpg')

            elif file_extension == 'mp4':
                detections = process_video(filepath)
                #return send_from_directory('static/runs/detect/mp4/', 'result.mp4')
                return send_file('static/runs/detect/mp4/result.mp4', mimetype='video/mp4')

    return render_template('index.html')


def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('static/runs/detect/mp4/result.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read() #cap.read==>membaca setiap frame video, mode iterasi
        if not ret:
            break
        detections = model.predict(frame)

        if detections and detections.prediction.bboxes_xyxy.any():
            annotated_frame = annotate_image(frame, detections)
        else:
            annotated_frame = frame
            #save_frame(annotated_frame, frame_number)

        frame_path = os.path.join('static', 'runs', 'detect', 'mp4', 'frame', f'frame_{frame_number:04d}.jpg')
        cv2.imwrite(frame_path, annotated_frame)
        frame_number+=1
        out.write(annotated_frame) #menambahkan hasil setiap frame ke result.mp4
    
    cap.release() #.release untuk memberihkan sumberdaya videocaptur dan writer dari memory
    out.release()
    return None

def annotate_image(image, detections):
    img = image.copy()
    '''bbox = detections.prediction.bboxes_xyxy[0]
    label = dataset_params['classes'][int(detections.prediction.labels[0])]
    confidence = float(detections.prediction.confidence[0])

    bbox_pt1 = (int(bbox[0]), int(bbox[1]))
    bbox_pt2 = (int(bbox[2]), int(bbox[3]))

    cv2.rectangle(img, bbox_pt1, bbox_pt2, (0, 255, 0), 1)
    cv2.rectangle(img, (bbox_pt1[0], bbox_pt1[1] - 20), (bbox_pt1[0] + 120, bbox_pt1[1]), (0, 255, 0), -1)
    cv2.putText(img, f"{label} {confidence:.2f}", (bbox_pt1[0], bbox_pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)'''

    for i, bbox in enumerate(detections.prediction.bboxes_xyxy):
        label = dataset_params['classes'][int(detections.prediction.labels[i])]
        confidence = float(detections.prediction.confidence[i])

        #mengubah gambar dalam format yang bisa diolah oleh OpenCV
        img = np.array(img, dtype=np.uint8)

        label_color, text_color = label_color_map.get(label, ((0, 0, 0), (255, 255, 255)))

        #mengubah format bounding box menjadi sesuai dengan cv2.rectangle
        bbox_pt1 = (int(bbox[0]), int(bbox[1]))
        bbox_pt2 = (int(bbox[2]), int(bbox[3]))

        cv2.rectangle(img, bbox_pt1, bbox_pt2, label_color, 1)
        cv2.rectangle(img, (bbox_pt1[0], bbox_pt1[1] - 20), (bbox_pt1[0] + 120, bbox_pt1[1]), label_color, -1)#color bacground text/dibuat rextangle
        cv2.putText(img, f"{label} {confidence:.2f}", (bbox_pt1[0], bbox_pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask App")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=False)