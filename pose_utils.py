import math
import numpy as np
from common_utils import OpenVinoModelWrapper
from typing import List, Tuple, Dict
import cv2

def calculate_iou_2(box1, box2):
    # Calculate the intersection coordinates
    [x1, y1, x2, y2] = box1
    [x3, y3 ,x4, y4] = box2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    # Calculate the area of intersection
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    # Calculate the areas of each box
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def convert_to_obtuse(angle):
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

def convert_to_acute(angle):
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    if angle > 90:
        angle = 180 - angle
    return angle

def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = sum([ba[i] * bc[i] for i in range(2)])
    norm_ba = math.sqrt(sum([x ** 2 for x in ba]))
    norm_bc = math.sqrt(sum([x ** 2 for x in bc]))
    cos_theta = dot_product / (norm_ba * norm_bc)
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def calculate_angles(keypoints):

    nose = keypoints.get(0)
    left_shoulder = keypoints.get(5)
    left_elbow = keypoints.get(7)
    left_palm = keypoints.get(9)
    right_shoulder = keypoints.get(6)
    right_elbow = keypoints.get(8)
    right_palm = keypoints.get(10)

    try:
        # Calculate angles for left elbow (ensure it's acute)
        angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_palm)
        angle_left_elbow = convert_to_obtuse(angle_left_elbow)
    except (TypeError, KeyError):
        angle_left_elbow = None

    try:
        # Calculate angles for right elbow (ensure it's acute)
        angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_palm)
        angle_right_elbow = convert_to_obtuse(angle_right_elbow)
    except (TypeError, KeyError):
        angle_right_elbow = None

    try:
        # Calculate angle for stretching out left hand horizontally (ensure it's acute)
        angle_left_hand_horizontal = convert_to_acute(math.degrees(math.atan2(left_palm[1] - left_shoulder[1], left_palm[0] - left_shoulder[0])))
    except (TypeError, KeyError):
        angle_left_hand_horizontal = None

    try:
        # Calculate angle for stretching out right hand horizontally (ensure it's acute)
        angle_right_hand_horizontal = convert_to_acute(math.degrees(math.atan2(right_palm[1] - right_shoulder[1], right_palm[0] - right_shoulder[0])))
    except (TypeError, KeyError):
        angle_right_hand_horizontal = None

    vulnerability = ""
    vul_id_arr = [False]*3

    if (angle_left_elbow is not None and angle_left_elbow < 120) or (angle_right_elbow is not None and angle_right_elbow < 120):
        vulnerability += "Possible Mobile Usage (Twisted Elbow), "
        vul_id_arr[0] = True


    # Depends on Camera angle

    if angle_left_hand_horizontal is not None and angle_left_hand_horizontal < 10:
        vulnerability += "Hands not on handle (Left Hand), "
        vul_id_arr[1] = True

    if angle_right_hand_horizontal is not None and angle_right_hand_horizontal < 10:
        vulnerability += "Hands not on handle (Right Hand), "
        vul_id_arr[2] = True

    return [angle_left_elbow, angle_right_elbow, angle_left_hand_horizontal, angle_right_hand_horizontal], vulnerability, vul_id_arr

def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    box_xyxy = box.copy()
    box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
    box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
    box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
    box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
    return box_xyxy

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    '''
    box and boxes are format as [x1, y1, x2, y2]
    '''
    # inter area
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, xmax-xmin) * np.maximum(0, ymax-ymin)

    # union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / union_area

def nms_process(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    sorted_idx = np.argsort(scores)[::-1]
    keep_idx = []
    while sorted_idx.size > 0:
        idx = sorted_idx[0]
        keep_idx.append(idx)
        ious = compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
        rest_idx = np.where(ious < iou_thr)[0]
        sorted_idx = sorted_idx[rest_idx+1]
    return keep_idx

class Yolov8KeypointsInference(object):
    ''' yolov8-keypoints onnxruntime inference
    '''

    def __init__(self,meta_obj, 
                 onnx_path: str,
                 input_size: Tuple[int],
                 box_score=0.25,
                 kpt_score=0.5,
                 nms_thr=0.2
                 ) -> None:
        self.model = OpenVinoModelWrapper(meta_obj, onnx_path)
        self.input_size = input_size
        self.box_score = box_score
        self.kpt_score = kpt_score
        self.nms_thr = nms_thr
        self.COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                   [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                   [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])

    def _preprocess(self, img: np.ndarray):
        ''' preprocess image for model inference
        '''
        input_w, input_h = self.input_size
        if len(img.shape) == 3:
            padded_img = np.ones((input_w, input_h, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114
        r = min(input_w / img.shape[0], input_h / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        # (H, W, C) BGR -> (C, H, W) RGB
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _postprocess(self, output: List[np.ndarray], ratio) -> Dict:

        """
        explaining ratio:
        consider orig image size = (1280, 6400)
        need to convert to 640, 640

        so the ratio is: 1/20, 1/100 -> taking min -> do 1/100

        """
        predict = output[0].squeeze(0).T                            ### (8400, 56)
        predict = predict[predict[:, 4] > self.box_score, :]        ### (69, 56)   consider in 69 cases, box prediction conf > required confidence
        scores = predict[:, 4]                                      ### (69, 1)
        boxes = predict[:, 0:4] / ratio                             ### (69, 4)
        boxes = xywh2xyxy(boxes)                                    
        kpts = predict[:, 5:]                                       ### (69, 51)
        # print(kpts.shape)
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:
                    kpts[i, 3*j: 3*(j+1)] = [-1, -1, -1]            ### dummy jinis coz of noob confidence
                else:
                    kpts[i, 3*j] /= ratio
                    kpts[i, 3*j+1] /= ratio
        idxes = nms_process(boxes, scores, self.nms_thr)            ### nms (15 indices)
        result = {
            'boxes':    boxes[idxes,: ].astype(int).tolist(),
            'kpts':     kpts[idxes,: ].astype(float).tolist(),
            'scores':   scores[idxes].tolist()
        }
        return result

    def detect(self, img: np.ndarray) -> Dict:
        img, ratio = self._preprocess(img)
        output = self.model(img[None, :]/255)
        """
            output shape -> 1, 56, 8400
            56 = 4 (box) + 1 (conf of box) + 17 (kpt) * 3 (x,y,conf)
        """
        # ort_input = {self.sess.get_inputs()[0].name: img[None, :]/255}
        # output = self.sess.run(None, ort_input)
        result = self._postprocess([output], ratio)
        return result


    def draw_result(self, orig_full_img, img: np.ndarray, result: Dict, with_label=False, base_coo=(0,0)):
        """
            result 
                -> boxes (15, 4)
                -> kpts (15, 51)
                -> scores (15, )
        """
        base_x0, base_y0 = base_coo
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        skeletons = []
        for box, kpt, score in zip(boxes, kpts, scores):

            filtered_pose = {}

            for idx in range(len(kpt) // 3):
                x, y, score = kpt[3*idx: 3*(idx+1)]
                x, y = x+base_x0, y+base_y0
                if score > 0:
                    cv2.circle(orig_full_img, (int(x), int(y)), 3, self.COLOR_LIST[idx], -1)
                    cv2.putText(orig_full_img,str(idx),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.3,[0, 255, 0],thickness=1)
                    if idx in [0,6,8,10,5,7,9]:
                      filtered_pose[idx] = (int(x), int(y));

            skeletons.append(filtered_pose)

            ### NECK TO SHOULDER LEFT
            try: cv2.line(img, filtered_pose[0], filtered_pose[6], (255,255,255), 3)
            except: pass

            ### SHOULDER TO ELBOW LEFT
            try: cv2.line(img, filtered_pose[6], filtered_pose[8], (255,255,255), 3)
            except: pass

            ### ELBOW TO WRIST LEFT
            try: cv2.line(img, filtered_pose[8], filtered_pose[10], (255,255,255), 3)
            except: pass

            ### NECK TO SHOULDER RIGHT
            try: cv2.line(img, filtered_pose[0], filtered_pose[5], (255,255,255), 3)
            except: pass

            ### SHOULDER TO ELBOW RIGHT
            try: cv2.line(img, filtered_pose[5], filtered_pose[7], (255,255,255), 5)
            except: pass

            ### ELBOW TO WRIST RIGHT
            try: cv2.line(img, filtered_pose[7], filtered_pose[9], (255,255,255), 5)
            except: pass

        return orig_full_img , skeletons

    def __call__(self, img, orig_full_img, base_coo):
      """
      Input : (h,w,3)
      output: annotated img (h,w,3) , skeleton dict
      """

      orig_full_img = np.copy(orig_full_img)
      result = self.detect(img)
      """
        result 
            -> boxes (15, 4)
            -> kpts (15, 51)
            -> scores (15, )
      """
      bounding_boxes_pose = result['boxes']
      img,skeletons = self.draw_result(orig_full_img, img, result, base_coo=base_coo)
      return img, skeletons, bounding_boxes_pose