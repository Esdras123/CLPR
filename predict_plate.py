import detectron2

import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import model.alpr as alpr

def predict_crnn(src):
  # create ALPR instance (change parameters according to needs)
  lpr = alpr.AutoLPR(decoder='bestPath', normalise=True)

  # load model (change parameters according to needs)
  lpr.load(crnn_path='./model/weights/best-fyp-improved.pth')

  # inferencing
  return lpr.predict(src)

def init_predictor():
  cfg = get_cfg()
  cfg.merge_from_file("../license plate/configs/lp_faster_rcnn_R_50_FPN_3x.yaml")
  cfg.MODEL.WEIGHTS = "./model_final.pth"
  cfg.MODEL.DEVICE = "cpu"
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
  predictor = DefaultPredictor(cfg)
  return predictor

def crop_plate(img):
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(gray,80,255, cv2.THRESH_BINARY)
  contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  areas = [cv2.contourArea(c) for c in contours]
  if(len(areas)!=0):
      max_index = np.argmax(areas)
      cnt=contours[max_index]
      x,y,w,h = cv2.boundingRect(cnt)
      bounds = cv2.boundingRect(cnt)
      secondCrop = img[y:y+h,x:x+w]
  else:
      secondCrop = img
  return secondCrop

def predict_plates(src):
  im = cv2.imread(src)
  predictor = init_predictor()
  outputs = predictor(im)
  box = outputs["instances"].get_fields()["pred_boxes"]

  plates = []
  for indexes in outputs["instances"].get_fields()["pred_boxes"].tensor.cpu().numpy():
    x1 = int(indexes[0])
    y = int(indexes[1])
    h = int(indexes[3]) - y
    w = int(indexes[2]) - x1

    if h>0:
      y1 = y
    else:
      y1 = y+h

    img = im[y1:y1+h, x1:x1 + w]
    img2 = crop_plate(img)

    #cv2.imshow("ddd", img)
    txt = predict_crnn(img)

    #to use
    if (txt != ''):
      plates.append(txt)
      cv2.rectangle(im, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 4)
      cv2.putText(im, txt, (x1, y1 - 15),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


  return im, plates

import sys
try:
  if len(sys.argv) > 1:
    img, plates = predict_plates(sys.argv[1])
    print("Prediction: ", plates)
    #cv2.imshow("Image of the car", img)
  else:
    print("Insert the path of the image as an argument")
except Exception:
  print("Error, please insert a correct path")
