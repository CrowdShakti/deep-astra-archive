# Copyright 2021 CrowdShakti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any
from deepface import detectors
import deepface

from fastapi import APIRouter, UploadFile, File, WebSocket, BackgroundTasks
from starlette.websockets import WebSocketDisconnect
from tensorflow.keras.preprocessing import image

from app.model.face_processor import Face
from app.service import test as test_service
from core.lib import util, logger
from core.model.handler import Resp

from deepface import DeepFace
from deepface.detectors import FaceDetector
import json
import os
import shutil
import cv2
import pandas as pd

LOGGER = logger.for_handler('face_recognition')

ROUTER = APIRouter()

'''
    'database' and 'photos' folders needs to be created in files under app
    faces.json
    {
        "0":{
            'number_of_photos': <int>,
            'number_of_faces': <int>,
            'max_id': <int>,
            'number_of_empty_ids': <int>,
            'empty_ids': <list>
        }
        "1":{
            'name': <str>,
            'photos': <list> # list of images this person is in
        }
    }
'''

@ROUTER.post('/api/v1/face-processor/faces')
def add_face(face_image: File, name: str, image: str):
    """
    Registers the face into the database.

    Args:
        face_image: Close up image of face
        name: name of person
        image: file name of the person's first image example (3.jpg)
    Returns:
        bool: True if successfulyy added else False
    """
    with open('../files/faces.json') as file_object:
        faces_data = json.load(file_object)
    if faces_data["0"]['number_of_empty_ids'] > 0:
        id_to_use = faces_data["0"]['empty_ids'].pop()
        faces_data["0"]['number_of_empty_ids'] -= 1
    else:
        id_to_use = faces_data["0"]['max_id'] + 1
        faces_data["0"]['max_id'] += 1
    database_path = 'C:\\Users\\Shubham\\Desktop\\internship\\face_recognition\\deep_astra\\deep-astra\\app\\files\\database'
    os.mkdir(database_path+'\\' + str(id_to_use))
    cv2.imwrite(database_path+'\\' + str(id_to_use) + '\\1.jpg', face_image)
    faces_data["0"]['number_of_faces'] += 1
    faces_data[str(id_to_use)] = dict()
    faces_data[str(id_to_use)]['name'] = name
    faces_data[str(id_to_use)]['photos'].append(image)
    with open('../files/faces.json', 'w') as file_object:
        json.dump(faces_data, file_object, intent = 4)
    return True




@ROUTER.delete('/api/v1/face-processor/faces')
def delete_face(id: int):
    """
    Deletes the face from the database.

    Args:
        id: int # it is the id of the face (to be deleted) in the database
    Returns:
        bool: True is successfully deleted otherwise False
    """
    if id <= 0:
        return False
    with open('../files/faces.json') as file_object:
        faces_data = json.load(file_object)
    database_path = 'C:\\Users\\Shubham\\Desktop\\internship\\face_recognition\\deep_astra\\deep-astra\\app\\files\\database'
    shutil.rmtree(database_path+'\\'+str(id))
    faces_data.pop(str(id))
    faces_data["0"]['number_of_faces'] -= 1
    faces_data["0"]['number_of_empty_ids'] += 1
    faces_data["0"]['empty_ids'].append(id)
    with open('../files/faces.json', 'w') as file_object:
        json.dump(faces_data, file_object, intent = 4)
    return True

    

@ROUTER.get('/api/v1/face-processor/faces')
def get_faces():
    """
    Retrieves all faces from the database.
    Returns:
        a list of lists
        [
            [image,name]
        ]
    """
    with open('../files/faces.json') as file_object:
        faces_data = json.load(file_object)
    list_of_faces = []
    database_path = 'C:\\Users\\Shubham\\Desktop\\internship\\face_recognition\\deep_astra\\deep-astra\\app\\files\\database'
    for id in faces_data:
        image = cv2.imread(database_path+'\\'+id+'\\1.jpg')
        list_of_faces.append([image, faces_data[id]['name']])
    return list_of_faces


@ROUTER.post('/api/v1/face-processor/predict')
def predict(image: File):
    """
    Predicts all the face from the image.
    Args:
        image (File): The image to be predicted.
    Returns:
        list of lists which contains encodings and locations of faces 
        and name of recognized face if known
        [
            [image,id,name,image_name] #if known face
            [image,image_name] #if unknown face
                                # id is index of that face in database
                                # image_name is name of the image stored in photos folder
        ]
    """
    with open('../files/faces.json') as file_object:
        faces_data = json.load(file_object)
    detector_model = 'opencv'
    database_path = 'C:\\Users\\Shubham\\Desktop\\internship\\face_recognition\\deep_astra\\deep-astra\\app\\files\\database'
    photos_path = 'C:\\Users\\Shubham\\Desktop\\internship\\face_recognition\\deep_astra\\deep-astra\\app\\files\\photos'
    cv2.imwrite(photos_path + '\\' + str(faces_data["0"]['number_of_photos'] + 1) + '.jpg', image)
    faces_data["0"]['number_of_photos'] += 1
    image_name = str(faces_data["0"]['number_of_photos']) + '.jpg'
    detector = FaceDetector.build_model(detector_model)
    faces = FaceDetector.detect_faces(detector,detector_model,image)
    number_of_faces = len(faces)
    list_of_faces = []
    for i in range(number_of_faces):
        result = pd.DataFrame()
        if len(os.listdir(database_path)) != 0:
            result = DeepFace.find(img_path = faces[i][0], db_path = database_path, model_name = 'Facenet')
        if result.empty:
            list_of_faces.append([faces[i][0], image_name])
        else:
            id = result.iloc[0,0].split('\\')[-1].split('/')[0]
            faces_data[id]['photos'].append(image_name)
            list_of_faces.append([faces[i][0], int(id), faces_data[id]['name'], image_name])
    with open('../files/faces.json', 'w') as file_object:
        json.dump(faces_data, file_object, intent = 4)
    return list_of_faces