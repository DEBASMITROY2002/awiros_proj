import cv2
import numpy as np
from fetch_data import *
from pose_utils import Yolov8KeypointsInference, calculate_iou_2, calculate_angles
from common_utils import OpenVinoModelWrapper


def process_yolo_outputs(
    out,
    conf_threshold,
    score_threshold,
    nms_threshold,
    x_scale,
    y_scale,
    class_filter_ids: list,
):
    scores = []
    class_ids = []
    boxes = []
    n_detections = out.shape[-1]

    class_wise_list = {}
    for c in class_filter_ids:
        class_wise_list[c] = {}
        class_wise_list[c]["boxes"] = []
        class_wise_list[c]["scores"] = []
        class_wise_list[c]["class_ids"] = []

    for i in range(n_detections):
        detect = out[:, i]
        class_score = detect[4:]
        class_ids_temp = np.argsort(class_score).tolist()[::-1][:2]

        for class_id in class_ids_temp:
            if (class_id in class_filter_ids) and class_score[
                class_id
            ] > score_threshold:
                class_wise_list[class_id]["scores"].append(class_score[class_id])
                class_wise_list[class_id]["class_ids"].append(class_id)
                x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                left = int((x - w / 2) * x_scale)
                top = int((y - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([left, top, left + width, top + height])
                class_wise_list[class_id]["boxes"].append(box)

    conf_is_none = conf_threshold is None

    for c in class_filter_ids:
        __sc = class_wise_list[c]["scores"]
        try:
            if conf_is_none and len(__sc) > 0:
                conf_threshold = np.median(__sc) + 0
                # print(np.median(__sc), np.mean(__sc))
        except:
            conf_threshold = 0.5

        indices = cv2.dnn.NMSBoxes(
            class_wise_list[c]["boxes"],
            np.array(class_wise_list[c]["scores"]),
            conf_threshold,
            nms_threshold=nms_threshold,
        )
        class_wise_list[c]["boxes"] = np.array(class_wise_list[c]["boxes"])[
            indices
        ].tolist()
        class_wise_list[c]["class_ids"] = np.array(class_wise_list[c]["class_ids"])[
            indices
        ].tolist()
        class_wise_list[c]["scores"] = np.array(class_wise_list[c]["scores"])[
            indices
        ].tolist()
        boxes.extend(class_wise_list[c]["boxes"])
        class_ids.extend(class_wise_list[c]["class_ids"])
        scores.extend(class_wise_list[c]["scores"])

    return scores, class_ids, boxes, class_wise_list


class RiderDetector:
    def __init__(self, meta_obj, yolo_model_path):
        self.classNames = ["person", "bicycle", "car", "motorbike"]
        self.model = OpenVinoModelWrapper(meta_obj, yolo_model_path)
        self.box_colors = [
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 0),
            (255, 0, 255),
        ]

    def calculate_iou(self, box1, box2):
        # Calculate the intersection coordinates
        [x1, y1, x2, y2] = box1
        [x3, y3, x4, y4] = box2
        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)

        # Calculate the area of intersection
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(
            0, intersection_y2 - intersection_y1
        )

        # Calculate the areas of each box
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        # Calculate the IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)

        return iou

    def find_matching_pairs(
        self,
        person_list,
        bike_list,
        iou_threshold,
        y_threshold,
        x_threshold,
        cg_gap_margin=20,
    ):
        matching_pairs = []

        # Create dictionaries to store the best matches
        best_person_for_bike = {}
        best_bike_for_person = {}

        for person_index, person_box in enumerate(person_list):
            person_box = person_box  # ["coor"]
            [person_x1, person_y1, person_x2, person_y2] = person_box
            person_center_x = (person_x1 + person_x2) / 2
            person_center_y = (person_y1 + person_y2) / 2

            best_iou = -1  # Initialize the best IoU score
            best_bike_index = -1  # Initialize the index of the best-matched bike

            for bike_index, bike_box in enumerate(bike_list):
                bike_box = bike_box  # ["coor"]
                [bike_x1, bike_y1, bike_x2, bike_y2] = bike_box
                bike_center_x = (bike_x1 + bike_x2) / 2
                bike_center_y = (bike_y1 + bike_y2) / 2

                iou = self.calculate_iou(person_box, bike_box)

                if (
                    iou > iou_threshold
                    and (person_center_y - cg_gap_margin) < bike_center_y
                    and abs(person_center_x - bike_center_x) < x_threshold
                ):
                    if iou > best_iou:
                        # Update the best match for the person
                        best_iou = iou
                        best_bike_index = bike_index

            # If a best match is found for the person, store it
            if best_iou > -1:
                best_person_for_bike[person_index] = best_bike_index

        # Create a list of matching pairs
        for person_index, bike_index in best_person_for_bike.items():
            matching_pairs.append(
                (
                    person_list[person_index],
                    bike_list[bike_index],
                    self.calculate_iou(
                        person_list[person_index], bike_list[bike_index]
                    ),
                )
            )

        return matching_pairs

    def fuse_bb(self, riders):
        riders_bbs = []
        for r in riders:
            person_coo, vehicl_coo = r[0], r[1]
            x1_fuse = min(person_coo[0], vehicl_coo[0])
            x2_fuse = max(person_coo[2], vehicl_coo[2])
            y1_fuse = min(person_coo[1], vehicl_coo[1])
            y2_fuse = max(person_coo[3], vehicl_coo[3])
            riders_bbs.append([x1_fuse, y1_fuse, x2_fuse, y2_fuse])
        return riders_bbs

    def __call__(self, orig_img, cv2_img, iou_threshold, y_threshold, x_threshold):
        """
        pil_img : cv2_img (_,_,3)
        returns:
        riders_bbs[i] = ith rider combo [((x1,y1),(x2,y2))]
        """
        x_scale = orig_img.shape[1] / 640
        y_scale = orig_img.shape[0] / 640

        img_ = cv2.resize(cv2_img, dsize=(640, 640))
        img = (img_.transpose((2, 0, 1))) / 255.0
        img = np.expand_dims(img, 0)

        output = self.model(img)[0]

        scores, class_ids, boxes, class_wise_list = process_yolo_outputs(
            output,
            conf_threshold=None,
            score_threshold=0.1,
            nms_threshold=0.8,
            x_scale=x_scale,
            y_scale=y_scale,
            class_filter_ids=[0, 1, 3],
        )

        persons = class_wise_list[0]["boxes"]
        bikes = class_wise_list[3]["boxes"]
        bicycles = class_wise_list[1]["boxes"]
        vehicles = bikes + bicycles
        # print(persons)
        riders = self.find_matching_pairs(
            persons, vehicles, iou_threshold, y_threshold, x_threshold
        )
        riders_bbs = self.fuse_bb(riders)
        img_np = np.copy(orig_img)


        img_rider_ann = img_np
        return persons, vehicles, riders, riders_bbs, img_rider_ann


class FullPipeline:
    def __init__(
        self,
        meta_obj,
        rider_onnx_path="/content/drive/MyDrive/helmet/yolov8_generic.onnx",
        hel_onnx_path="/content/drive/MyDrive/helmet/hel0_model.onnx",
        anr_onnx_path="/content/drive/MyDrive/helmet/anr_model.onnx",
        pose_model_path="/content/drive/MyDrive/helmet/yolov8m-pose.onnx",
    ):
        self.rider_model = RiderDetector(meta_obj, rider_onnx_path)
        self.hel_model = OpenVinoModelWrapper(meta_obj, hel_onnx_path)
        self.anr_model = OpenVinoModelWrapper(meta_obj, anr_onnx_path)
        self.pose_model = Yolov8KeypointsInference(
            meta_obj, pose_model_path, (640, 640), 0.7, 0.8, 0.7
        )
        self.warning_color = (255, 0, 0)
        self.warning_boldness = 2
        self.warning_size = 1

        self.sev_color = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]

    def preprocess_img_for_vgg(self, img, taget_shape=(64, 64)):
        blob = cv2.resize(img, dsize=taget_shape, interpolation=cv2.INTER_CUBIC)
        blob = np.expand_dims(blob, 0)
        return blob / 255

    def preprocess_img_for_yolo(self, img, taget_shape=(640, 640)):
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255, taget_shape, swapRB=True, mean=(0, 0, 0), crop=False
        )
        return blob

    def __call__(self, orig_img):
        """
        Input: (_,_,3)
        """
        src_img_640_640 = cv2.resize(np.copy(orig_img), dsize=(640, 640))
        _, _, riders, riders_bbs_xyxy, img_rider_ann = self.rider_model(
            orig_img, src_img_640_640, iou_threshold=0.10, y_threshold=0, x_threshold=30
        )
                
        ### riders : list<person, bike, iou>
        ### riders_bbs_xyxy: coordinates (xmin, ymin, xmax, ymax) from fuse_bb
        ### img_rider_ann: original image + annotated riders

        blobs = []

        for i, [x0, y0, x1, y1] in enumerate(riders_bbs_xyxy):
            ### Uncomment In App
            blb = blob()
            blb.tx = x0
            blb.ty = y0
            blb.bx = x1
            blb.by = y1
            blb.cropped_frame = orig_img[blb.ty : blb.by, blb.tx : blb.bx, :]
            blb.id = i
            blb.conf = round(float(96), 3)
            blb.attribs["PERSON"] = "Person"+str(i)
            blb.label = str(i)+" -> "

            #### finding heads 
            head_crop = orig_img[
                y0 : (y0 + (y1 - y0) // 4), 
                x0 + (x1 - x0) // 4 : x1 - (x1 - x0) // 4
            ]
            #### reshaping head
            preprocessed_head_crop = self.preprocess_img_for_vgg(
                head_crop, taget_shape=(64, 64)
            )
            #### predicting head -> helmet or not
            head_pred = (
                self.hel_model(preprocessed_head_crop)[0][0] < 0.3
            ) 

            """
            head_pred -> with or without helmet
            """

            ### Uncomment In App
            blb.attribs["HELMET"] = "OK" if head_pred else "TR"

            print("HELMET: " + blb.attribs["HELMET"])
            
            blobs.append(blb)

        
        #### Running pose estimation on whole image
        img_rider_ann, skeletons, bounding_boxes_pose = self.pose_model(
            np.copy(orig_img), img_rider_ann, base_coo=(0, 0)
        )
        ### img_rider_ann: annotated image + rider + skeleton
        ### skeleton: (number of person, count of necessary keypoints)
        ### bounding_boxes_pose: coordinates (xmin, ymin, xmax, ymax) of persons (may not be rider in this case)

        masked_orig_img = np.copy(orig_img)

        preprocessed_orig_img = self.preprocess_img_for_yolo(masked_orig_img,taget_shape = (640, 640))
        anr_result = self.anr_model(preprocessed_orig_img)[0]
        _, _, anr_boxes_xyxy, _ = process_yolo_outputs(
            anr_result, conf_threshold = None,
            score_threshold = 0.001 ,
            nms_threshold=0.80,
            x_scale = np.copy(orig_img).shape[1]/640,
            y_scale = np.copy(orig_img).shape[0]/640,
            class_filter_ids=[0])

        body_to_skel_mapping= {}
        body_to_anr_mapping = {}
        
        for i,[x0,y0,x1,y1] in enumerate(riders_bbs_xyxy):
            body_to_skel_mapping[i] = -1
            body_to_anr_mapping[i] = -1

            # Mapping skel
            max_iou = 0
            max_iou_idx = None
            for j,[sx0,sy0,sx1,sy1] in enumerate(bounding_boxes_pose):
                iou = calculate_iou_2([x0,y0,x1,y1], [sx0,sy0,sx1,sy1])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j
            if max_iou >= 0.1:
                body_to_skel_mapping[i] = max_iou_idx

            # Mapping anr
            max_iou = 0
            max_iou_idx = None
            for j,[ax0,ay0,ax1,ay1] in enumerate(anr_boxes_xyxy):
                iou = calculate_iou_2([x0,y0,x1,y1], [ax0,ay0,ax1,ay1])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j
            if max_iou >= 0.001:
                body_to_anr_mapping[i] = max_iou_idx

        print(len(skeletons))
        print(body_to_skel_mapping)

        useless_idx = []
        severity_level = 0

        for b_id, s_id in body_to_skel_mapping.items():
            n_id = body_to_anr_mapping[b_id]
            if s_id == -1:
                vul_id_arr = [0, 0, 0]
            
            else:
                angles, vulnerability, vul_id_arr = calculate_angles(skeletons[s_id])
                print(
                    f"Angle for left elbow (acute): {angles[0]}\nAngle for right elbow (acute): {angles[1]}\nAngle for stretching out left hand horizontally: {angles[2]}\nAngle for stretching out right hand horizontally: {angles[3]}\nVulnerability: {vulnerability}"
                )

            if n_id != -1:
                cv2.rectangle(
                    img_rider_ann,
                    (anr_boxes_xyxy[n_id][0], anr_boxes_xyxy[n_id][1]),
                    (anr_boxes_xyxy[n_id][2], anr_boxes_xyxy[n_id][3]),
                    [255]*3,
                    3
                )

            curr_severity_level = 0
            
            
            if not (vul_id_arr[0] or vul_id_arr[1] or vul_id_arr[2] or blobs[b_id].attribs["HELMET"] == "TR"):
                useless_idx.append(b_id)
                cv2.rectangle(img_rider_ann, (riders_bbs_xyxy[b_id][0], riders_bbs_xyxy[b_id][1]), (riders_bbs_xyxy[b_id][2], riders_bbs_xyxy[b_id][3]), self.sev_color[0], 2)
            else:
                blobs[b_id].attribs["MOBILE"] = "TR" if vul_id_arr[0] else "OK"
                blobs[b_id].attribs["LEFT"] = "TR" if vul_id_arr[1] else "OK"
                blobs[b_id].attribs["RIGHT"] = "TR" if vul_id_arr[2] else "OK"
                if blobs[b_id].attribs["HELMET"] == "TR": 
                    blobs[b_id].label += "Hel,"
                    img_text += "H"
                    curr_severity_level = 2
                if blobs[b_id].attribs["MOBILE"] == "TR":
                    blobs[b_id].label += "Mob,"
                    img_text += "M"
                    if curr_severity_level == 2:
                        curr_severity_level = 3
                    curr_severity_level = max(curr_severity_level, 1)
                if blobs[b_id].attribs["LEFT"] == "TR" or blobs[b_id].attribs["RIGHT"] == "TR": 
                    blobs[b_id].label += "HFree,"
                    img_text += "F"
                    if curr_severity_level == 2:
                        curr_severity_level = 3
                    curr_severity_level = max(curr_severity_level, 1)
                cv2.rectangle(img_rider_ann, (riders_bbs_xyxy[b_id][0], riders_bbs_xyxy[b_id][1]), (riders_bbs_xyxy[b_id][2], riders_bbs_xyxy[b_id][3]), self.sev_color[curr_severity_level], 2)
                cv2.putText(
                    img_rider_ann,
                    img_text,
                    (riders_bbs_xyxy[b_id][0], riders_bbs_xyxy[b_id][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    thickness=self.warning_boldness,
                )
            

                
        fin_blobs = []
        for idx, elem in enumerate(blobs):
            if not idx in useless_idx:
                fin_blobs.append(elem)

        return img_rider_ann, fin_blobs, severity_level