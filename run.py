

from fetch_data import *
import sys
import cv2
import random

import openvino
from openvino.runtime import Core
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.runtime import Layout, AsyncInferQueue, PartialShape
import numpy as np
from yolov7 import *

import model_utils

classes = ['rider']

def run(acs_url, broker_url, topic_name):


    print("Creating Meta....")
    meta_obj = meta(acs_url, broker_url, topic_name)

    fp = model_utils.FullPipeline(
        meta_obj,
        '/home/awiros-docker/alexandria/securus.generic/openvino/fp16/yolov8_generic.xml', 
        '/home/awiros-docker/alexandria/securus.vgg19/openvino/fp16/hel_vgg_model.xml', 
        '/home/awiros-docker/alexandria/securus.anr/openvino/fp16/anr_model.xml',
        '/home/awiros-docker/alexandria/securus.pose/openvino/fp16/yolov8-pose.xml')

    sev_list = ["awi_low", "awi_medium", "awi_high", "awi_critical"]

    print("Parsing Acs....")
    meta_obj.parse_acs()
    # exit()
    
    # matha_model = cv2.dnn.readNetFromONNX("/home/awiros-docker/alexandria/securus2.yolov8/onnx/hel_vgg_model.onnx")
    while True:
        print("Getting Frame....")
        meta_obj.run_camera()
        for stream_index,stream in enumerate(meta_obj.streams):

            out_image, blobs, severity = fp(stream)

            eve = event()

            frame_h,frame_w,_ = stream.shape

            """
            WRITE YOUR LOGIC OVER HERE
            """
            print("LEN: " + str(len(blobs)))
            if (len(blobs)>0):
                for blb in blobs:
                    # print(blb.label + " Detected!!")

                    # if (blb.label!="rider"):
                    #     continue
                    
                    if blb.cropped_frame.shape[0] == 0 or blb.cropped_frame.shape[1] == 0:
                        continue

                    blb.frame = stream                    
                    eve.eve_blobs.append(blb)
                    print(f"cropped_frame: {blb.cropped_frame.shape}")


                eve.set_frame(out_image)
                #cv2.imwrite("out.jpg",out_image)
                eve.type = "Test2"
                eve.severity = sev_list[severity]
                print("SEV: " + sev_list[severity])
                event.source_type_key = "Test3"
                event.source_entity_idx = random.randint(0,1000)
                meta_obj.push_event(eve)

                print("Pushing alert....")
                meta_obj.send_event()        


if __name__ == "__main__":

    acs_url = sys.argv[1]
    broker_url = sys.argv[2]
    topic_name = sys.argv[3]

    run(acs_url, broker_url, topic_name)




