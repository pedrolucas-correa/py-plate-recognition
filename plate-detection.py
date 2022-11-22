#%% Imports necessários 

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import cv2
import pandas as pd
pd.options.mode.chained_assignment = None 

import glob
import xml.etree.ElementTree as ET
import splitfolders
import easyocr
import yaml

from pathlib import Path
from timeit import default_timer as timer

import torch

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

#%% Dicionário com as informações básicas do dataset

dataset = {
            "file":[],
            "width":[],
            "height":[],
            "xmin":[],
            "ymin":[],
            "xmax":[],
            "ymax":[]
           }

#%% Exploramos as imagens no dataset e separamos as imagens .png .jpg e seus respectivos documentos .xml em duas listas

img_names=[] 
annotations=[]
for dirname, _, filenames in os.walk("./car-plates"):
    for filename in filenames:
        if os.path.join(dirname, filename)[-3:]==("png" or "jpg"):
            img_names.append(filename)
        elif os.path.join(dirname, filename)[-3:]=="xml":
            annotations.append(filename)
    
img_names[:10]

#%%

annotations[:10]

#%% Reescrevemos os .xml para nosso dicionário

path_annotations="./car-plates/annotations/*.xml" 

for item in glob.glob(path_annotations):
    tree = ET.parse(item)
    
    for elem in tree.iter():
        if 'filename' in elem.tag:
            filename=elem.text
        elif 'width' in elem.tag:
            width=int(elem.text)
        elif 'height' in elem.tag:
            height=int(elem.text)
        elif 'xmin' in elem.tag:
            xmin=int(elem.text)
        elif 'ymin' in elem.tag:
            ymin=int(elem.text)
        elif 'xmax' in elem.tag:
            xmax=int(elem.text)
        elif 'ymax' in elem.tag:
            ymax=int(elem.text)
            
            dataset['file'].append(filename)
            dataset['width'].append(width)
            dataset['height'].append(height)
            dataset['xmin'].append(xmin)
            dataset['ymin'].append(ymin)
            dataset['xmax'].append(xmax)
            dataset['ymax'].append(ymax)
        
classes = ['license']

# %%

df=pd.DataFrame(dataset)
df

# %%

df.info()

#%% Normalizando os dados para uso do YOLOv5 em arquivos .txt

x_pos = []
y_pos = []
frame_width = []
frame_height = []

labels_path = Path("car-plates/labels")

labels_path.mkdir(parents=True, exist_ok=True)

save_type = 'w'

for i, row in enumerate(df.iloc):
    current_filename = str(row.file[:-4])
    
    width, height, xmin, ymin, xmax, ymax = list(df.iloc[i][-6:])
    
    x=(xmin+xmax)/2/width
    y=(ymin+ymax)/2/height
    width=(xmax-xmin)/width
    height=(ymax-ymin)/height
    
    x_pos.append(x)
    y_pos.append(y)
    frame_width.append(width)
    frame_height.append(height)
    
    txt = '0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n'
    
    if i > 0:
        previous_filename = str(df.file[i-1][:-4])
        save_type='a+' if current_filename == previous_filename else 'w'

    with open("car-plates/labels/" + str(row.file[:-4]) +'.txt', save_type) as f:
        f.write(txt)
        
        
df['x_pos']=x_pos
df['y_pos']=y_pos
df['frame_width']=frame_width
df['frame_height']=frame_height

df

#%% Usando a biblioteca Splitfolders separamos as imagens e labels em sets de treinamento e de validação 

pasta_de_entrada = Path("./car-plates")
pasta_de_saida = Path("yolov5/data/plate-recognition")

splitfolders.ratio(
    pasta_de_entrada,
    output=pasta_de_saida,
    seed=42,
    ratio=(0.8, 0.2),
    group_prefix=None
)
print("Separação dos arquivos finalizada")

# %%

def caminhando_pela_pasta(dir_path: Path) -> None:
    """Imprime o conteudo das pastas"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"Tem {len(dirnames)} pastas e {len(filenames)} arquivos na pasta '{dirpath}'")

    
caminhando_pela_pasta(pasta_de_entrada)
print()
caminhando_pela_pasta(pasta_de_saida)

#%% Configuração do arquivo .yaml

yaml_file = 'yolov5/data/plates.yaml'

yaml_data = dict(
    path = "data/plate-recognition",
    train = "train",
    val = "val",
    nc = len(classes),
    names = classes
)

with open(yaml_file, 'w') as f:
    yaml.dump(yaml_data, f, explicit_start = True, default_flow_style = False)

#%% Caso não haja GPU, irá utilizar CPU como dispositivo

device = '0' if torch.cuda.is_available() else 'cpu' 
device

# %% Treino do modelo YOLOv5

start_time = timer()

# !cd yolov5 && python train.py --workers 2 --img 640 --batch 16 --epochs 3 --data "data/plates.yaml" --weights yolov5s.pt --device {device} --cache

end_time = timer()

print(f'Training time: {(end_time-start_time):.2f}')

#%% Definindo o modelo a ser usado

model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'yolov5/runs/train/exp5/weights/best.pt', force_reload=True)

reader = easyocr.Reader(['en'])

#%% Funções para detecção/leitura de placas em vídeo

def get_plates_xy(frame: np.ndarray, labels: list, row: list, width: int, height: int, reader: easyocr.Reader) -> tuple:
    '''Obtem os resultados do EasyOCR e retorna-os com as coordenadas da caixa delimitadora'''
    
    x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height) ## BBOx coordniates
    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ocr_result = reader.readtext(np.asarray(plate_crop), allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')#, paragraph="True", min_size=50)
    
    return ocr_result, x1, y1


def detect_text(i: int, row: list, x1: int, y1: int, ocr_result: list, detections: list, yolo_detection_prob: float=0.3) -> list:
    '''Verifica as probabilidades de detecção, descarta aqueles com pouca e reescreve a saída do ocr_reader para a lista detections'''
    
    if row[4] >= yolo_detection_prob: #discard predictions below the value             
        if(len(ocr_result))>0:
            for item in ocr_result:     
                    detections[i][0]=item[1]
                    detections[i][1]=[x1, y1]
                    detections[i][2]=item[2]
                    
    return detections

def is_adjacent(coord1: list, coord2: list) -> bool:
    '''Verifica se as cordenadas da primeira lista COORD1 é semelhante a lista COORD2'''
    
    MAX_PIXELS_DIFF=50
    
    if (abs(coord1[0] - coord2[0]) <= MAX_PIXELS_DIFF) and (abs(coord1[1] - coord2[1]) <= MAX_PIXELS_DIFF):
        return True
    else:
        return False
    

def sort_detections(detections: list, plates_data: list) -> list:
    '''Verifica as detecções do ultimo frame e reescreve os indexes para coordenadas similares'''
    
    for m in range(0, len(detections)):
        for n in range(0, len(plates_data)):
            if not detections[m][1]==[0, 0] and not plates_data[n][1]==[0,0]:
                if is_adjacent(detections[m][1], plates_data[n][1]):
                    if m!=n:
                        temp=detections[m]
                        detections[m]=detections[n]
                        detections[n]=temp
                        
    return detections
    
def delete_old_labels(detections: list, count_empty_labels: list, plates_data: list, frames_to_reset: int=3) -> tuple:
    '''Deleta placas que já não são detectadas no frame atual'''
    
    for m in range(0, len(detections)):
        if detections[m][0]=='None' and not count_empty_labels[m]==frames_to_reset:
            count_empty_labels[m]+=1
        elif count_empty_labels[m]==frames_to_reset:
            count_empty_labels[m]=0
            plates_data[m]=['None', [0,0], 0]
        else:
            count_empty_labels[m]=0
            
    return plates_data, count_empty_labels


def overwrite_plates_data(detections: list, plates_data: list, plate_lenght=None) -> list:
    '''Checa as coordenadas da lista detections, se houver registro semelhante em plate_data, tenta sobrescrevê-lo (apenas se a probabilidade for maior)'''
    
    if (detections[i][2]>plates_data[i][2] or detections[i][2]==0):
        if plate_lenght:
            if len(detections[i][0])==plate_lenght:
                plates_data[i][0]=detections[i][0]
                plates_data[i][2]=detections[i][2]       
        else:
            plates_data[i][0]=detections[i][0]
            plates_data[i][2]=detections[i][2]
    plates_data[i][1]=detections[i][1]
        
    return plates_data

#%% Resultado em vídeo

video_path = "./car-plates/cars-hd.mp4"
cap = cv2.VideoCapture(video_path)

plates_data = [['None', [0,0], 0] for n in range(5)]
count_empty_labels=[0]*5

assert cap.isOpened()

while(cap.isOpened()):
    ret, frame = cap.read()
    assert not isinstance(frame,type(None)), 'frame not found'
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    
    results = model(frame)   
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    width, height = frame.shape[1], frame.shape[0]
    
    detections=[['None', [0,0], 0] for n in range(5)]
    i=0 
    
    
    ## Lê todas as placas detectadas em cada frame e salva em detections
    while i < len(labels):    
        row = coordinates[i]
        ## Corta detecções e manda para o OCR
        ocr_result, x1, y1=get_plates_xy(frame, labels, row, width, height, reader)  
        
        ## Se prepara para cada frame
        detections=detect_text(i, row, x1, y1, ocr_result, detections, 0.5)
        i+=1    
    i=0
    
    ## Ordena as detecções 
    detections=sort_detections(detections, plates_data)
    
    ## Deleta dados de placas que desapareceram
    plates_data, count_empty_labels=delete_old_labels(detections, count_empty_labels, plates_data, 3)
            
    ## Reescreve os dados e imprime as predições de texto por cima das caixas selecionadas
    while i < len(labels):
        plates_data=overwrite_plates_data(detections, plates_data, 7)
        cv2.putText(frame, f"{plates_data[i][0]}", (plates_data[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        i+=1
    
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

#%% Resultado em imagem

%matplotlib inline

test_photo_path = "car-plates/carro-1.jpeg"

results = model(test_photo_path)
detections=np.squeeze(results.render())

labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
image = cv2.imread(test_photo_path)
width, height = image.shape[1], image.shape[0]

print(f'Photo width,height: {width},{height}. Detected plates: {len(labels)}')

for i in range(len(labels)):
    row = coordinates[i]
    if row[4] >= 0.6:
        x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
        plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
        ocr_result = reader.readtext((plate_crop), paragraph="True", min_size=120, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        text=ocr_result[0][1]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6) ## BBox
        cv2.putText(image, f"{text}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        plt.axis(False)
        plt.imshow((image)[...,::-1])
        
        print(f'Detection: {i+1}. YOLOv5 prob: {row[4]:.2f}, easyOCR results: {ocr_result}')
