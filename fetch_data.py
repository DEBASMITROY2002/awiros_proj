import urllib.request
import json 
import numpy as np
import cv2
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
import random
import base64
import datetime

import openvino
from openvino.runtime import Core
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.runtime import Layout, AsyncInferQueue, PartialShape



class meta:
    counter = 0
    def __init__(self,acs_url, broker_url, topic_name):
        self.url = acs_url
        self.awi_recog_conf = None
        self.awi_fuzzyness = None
        self.awi_sharpness = None
        self.awi_space_senstivity = None
        self.awi_time_senstivity = None
        self.awi_time_frequency = None
        self.awi_detection_conf = None

        self.awi_lines = None
        self.awi_coords = None 

        self.stream_url = None
        self.cap_list = None
        self.streams = None
        self.events = None
        self.frame_width = 1920
        self.frame_height = 1080 
        self.acs_data = None
        self.producer = KafkaProducer(bootstrap_servers=broker_url, max_request_size=3173440261)
        self.topic_name = topic_name

    def convert_to_unix(self):
        presentDate = datetime.datetime.now()
        unix_timestamp = datetime.datetime.timestamp(presentDate)*1000
        return unix_timestamp

    def create_openvino_model(self, model_path):
        print("Creating Openvino Model....")
        ie = Core()
        devices = ie.available_devices
        for device in devices:
            device_name = ie.get_property(device, "FULL_DEVICE_NAME")
            print(f"{device}: {device_name}")

        model =  ie.read_model(model_path)

        compiled_model = ie.compile_model(model=model, device_name=device)
        infer_request = compiled_model.create_infer_request()
        return model, compiled_model, infer_request

    def get_utc_time(self):
        current_datetime = datetime.datetime.utcnow()
        # Format it as per your requirement
        formatted_datetime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return formatted_datetime

    def parse_acs(self):
        with urllib.request.urlopen(self.url) as url:
            data = json.loads(url.read().decode())["awi_acs_list"]
            json_data = data
            print(data)
            self.acs_data = data
            self.awi_recog_conf = [x["awi_app"]["awi_config"]["awi_recog"]["awi_conf"] for x in json_data]
            self.awi_fuzzyness = [x["awi_app"]["awi_config"]["awi_detect"]["awi_fuzzyness"] for x in json_data]
            self.awi_sharpness = [x["awi_app"]["awi_config"]["awi_detect"]["awi_sharpness"] for x in json_data]
            self.awi_detection_conf = [x["awi_app"]["awi_config"]["awi_detect"]["awi_conf"] for x in json_data]

            
            self.awi_space_senstivity = [x["awi_app"]["awi_config"]["awi_app_param"]["awi_space_senstivity"] for x in json_data]
            self.awi_time_senstivity = [x["awi_app"]["awi_config"]["awi_app_param"]["awi_time_senstivity"] for x in json_data]
            self.awi_time_frequency = [x["awi_app"]["awi_config"]["awi_app_param"]["awi_time_frequency"] for x in json_data]
            
            self.awi_lines = [[[coord_dict for coord_dict in y["awi_coords"]]for y in x["awi_app"]["awi_config"]["awi_lines"]] for x in json_data ]

            # print("self awi lines: ",self.awi_lines)
            self.awi_regions = [[[coord_dict for coord_dict in y["awi_coords"]]for y in x["awi_app"]["awi_config"]["awi_regions"]] for x in json_data ]
            
            # print("self awi regions: ",self.awi_regions)
            print("Stream URL- ", json_data[0]["awi_source"]["awi_feed"]["awi_url"])
            self.stream_url = [x["awi_source"]["awi_feed"]["awi_url"] for x in json_data] #Handle for multiple streams!!
            
            
            self.frame_height = [x["awi_source"]["awi_feed"]["awi_params"]["awi_resolution"]["awi_height"] for x in json_data] #Handle for multiple streams!!
            self.frame_width = [x["awi_source"]["awi_feed"]["awi_params"]["awi_resolution"]["awi_width"] for x in json_data] #Handle for multiple streams!!
            self.cap_list = [cv2.VideoCapture(x) for x in self.stream_url]
            self.streams = [None]*len(self.stream_url)
            # self.events = [None]*(len(self.stream_url))
            self.events = []

    def run_camera(self):
        for stream_index,cap in enumerate(self.cap_list):
            success,frame = cap.read()
            print("Stream - ", success)
            self.streams[stream_index] = frame

    def push_event(self,event):
        self.events.append(event)

    def send_event(self):
        # print("Events inside send event: ",self.events)
        for i,event in enumerate(self.events):
            event.event_audit()     
            ads = self.acs_data[i]
            """
            Update the following:
            1)awi_id
            2)awi_url
            """
            ads["awi_response"] = {
                "awi_app" : {
                    "awi_id":0,
                    "awi_label":"App SO"
                },
                "awi_event": {
                    "awi_blobs":[],
                    "awi_engyn_timestamp":self.get_utc_time(),
                    "awi_frame":{
                        "awi_url":event.img_url
                    },
                    "awi_label":"",
                    "awi_latitude":"",
                    "awi_longitude":"",
                    "awi_severity":event.severity,
                    "awi_stream_timestamp":self.get_utc_time(),
                    "awi_timestamp":self.convert_to_unix(),
                    "awi_type":""
                },
                "awi_response_type":"app_event"
            }       
            # with open(f"./img_txt/{self.counter}_event.txt","w") as f:
            #     f.write(event.img_url)

            for blb_index,blb in enumerate(event.eve_blobs):
                blb.fill_norm_coords()
                if (blb.cropped_frame is not None):
                    blb.convert_to_b64(blb.cropped_frame)
                awi_coord = [
                    {
                        "x":blb.tx,
                        "y":blb.ty
                    },
                    {
                        "x":blb.bx,
                        "y":blb.ty
                    },
                    {
                        "x":blb.bx,
                        "y":blb.by
                    },
                    {
                        "x":blb.tx,
                        "y":blb.by
                    }
                ]
                awi_db = [
                    {
                        'awi_class': '',
                        'awi_conf': blb.conf,
                        'awi_label': blb.label,
                        'awi_severity': "",
                        'awi_subclass': '',
                        'id': blb.id,
                        'img_url': [''],
                        'imgs': ['']
                    }
                ]
                awi_final_dict = {
                    "awi_attribs":blb.attribs,
                    "awi_coords":awi_coord,
                    "awi_db":awi_db,
                    "awi_descriptor":[],
                    "awi_detection_time":self.convert_to_unix(),
                    "awi_label":blb.label,
                    "awi_orig_label":"",
                    "awi_score":blb.conf,
                    "awi_severity":"",
                    "awi_subclass":"",
                    "awi_type":"",
                    "awi_url":blb.crop_frame_url,
                    "awi_camid": blb.crop_frame_url,
                    "awi_image_url": blb.crop_frame_url,
                    "awi_guid": blb.crop_frame_url,
                    "awi_secondary_image_url":blb.crop_frame_url,
                }
                # ads["awi_response"]["awi_event"]["awi_blobs"].append({"awi_attribs":"","awi_coords":[{"x":blb.tx_norm,"y":blb.ty_norm},{"x":blb.bx_norm,"y":blb.ty_norm},{"x":blb.bx_norm,"y":blb.by_norm},{"x":blb.tx_norm,"y":blb.by_norm}]})
                ads["awi_response"]["awi_event"]["awi_blobs"].append(awi_final_dict)


            output = json.dumps(ads)
            future = self.producer.send(self.topic_name , value= bytes(output, 'utf-8'))
            try:
                record_metadata = future.get(timeout=50)
            except Exception as e:
                print("Error: ",e)
                # Decide what to do if produce request failed...
                pass
            """
            Send the event over here
            """
        self.events.clear()
        self.counter+=1

class blob:
    def __init__(self):
        self.bx = 100
        self.by = 100
        self.tx = 25
        self.ty = 25
        self.frame = np.zeros([480,640,3])
        self.id = random.randint(0,1000000)
        self.cropped_frame = None
        self.label = None
        self.conf = None
        self.bx_norm = None
        self.by_norm = None
        self.tx_norm = None
        self.ty_norm = None
        self.label = None
        self.crop_frame_url = None
        self.attribs = {}

    def draw_on_frame(self):
        return cv2.rectangle(self.frame, (self.tx,self.ty), (self.bx,self,self.by), (255,0,0), thickness=2)

    def get_blob_frame(self):
        cropped_frame = self.frame[self.ty:self.by,self.tx:self.bx,:]
        
        return cropped_frame

    def fill_norm_coords(self):

        self.tx_norm = self.tx/self.frame.shape[1]
        self.ty_norm = self.ty/self.frame.shape[0]
        self.bx_norm = self.bx/self.frame.shape[1]
        self.by_norm = self.by/self.frame.shape[0]

    def convert_to_b64(self,img):
        # img = cv2.resize(img,(640,480))
        retval, buffer = cv2.imencode('.jpg', img)
        self.crop_frame_url = base64.b64encode(buffer)
        self.crop_frame_url = str(self.crop_frame_url)[2:-1]



class event:
    def __init__(self):
        self.type = ""
        self.label = "" 
        self.timestamp = datetime.datetime.now()
        self.severity = "" 
        self.source_entity_idx = None    
        self.eve_frame = None
        self.eve_blobs = []
        self.eve_motion = None
        self.img_url = None
    
    def set_frame(self,frame):
        self.eve_frame = cv2.resize(frame,(640,480))
        retval, buffer = cv2.imencode('.jpg', frame)
        self.img_url = base64.b64encode(buffer)
        self.img_url = str(self.img_url)[2:-1]


        
    def event_audit(self):
        if(len(self.eve_blobs) == 0):
             raise Exception("No blobs found in the event!")
        if(self.eve_frame is None):
            raise Exception("Event frame not set!")
            
        # del self.eve_blobs





    

