import cv2
import numpy as np
import math
import argparse


class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i]))
                         for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # if isinstance(outputs, tuple):
        #     outputs = list(outputs)
        # if float(cv2.__version__[:3])>=4.7:
        #     outputs = [outputs[2], outputs[0], outputs[1]] ###opencv4.7需要这一步，opencv4.5不需要
        # Perform inference on the image
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))  ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred,
                                      max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

class face_quality_assessment():
    def __init__(self, path):
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 112
        self.input_width = 112

    def detect(self, srcimg):
        input_img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        input_img = (input_img.astype(np.float32) / 255.0 - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(input_img.astype(np.float32))
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs[0].reshape(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/1.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    face_detector = YOLOv8_face("weights/yolov8n-face.onnx", conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    fqa = face_quality_assessment("weights/face-quality-assessment.onnx")

    srcimg = cv2.imread(args.imgpath)
    # Detect Objects
    boxes, scores, classids, kpts = face_detector.detect(srcimg)
    drawimg = srcimg.copy()
    for i, box in enumerate(boxes):
        x, y, w, h = box.astype(int)
        crop_img = srcimg[y:y + h, x:x + w]  ###也可以在人脸矫正之后做crop
        fqa_probs = fqa.detect(crop_img)
        fqa_prob_mean = round(np.mean(fqa_probs), 2)

        cv2.rectangle(drawimg, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        cv2.putText(drawimg, "fqa_score:"+str(fqa_prob_mean), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)

    # cv2.imwrite('result.jpg', drawimg)
    winName = 'Deep learning face-quality-assessment use OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()