import os 
import numpy as np 
import random
from PIL import Image
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb

classes1 = ['fire','person','car','smoke','candlelight']

sets = ['train','test','val']



def preprocessTrueBox(trueBoxes,inputShape,anchors,numClasses):
  

    #print(trueBoxes[0][...,:1],trueBoxes[0][...,1:])
    #trueBoxes = np.concatenate((trueBoxes[:][...,1:],trueBoxes[:][...,:1]),axis=1)
    #trueBoxes = np.expand_dims(trueBoxes,axis=0)
    assert(trueBoxes[...,0]<numClasses).all()
    numLayers = len(anchors)//3
    anchorMask = [[6,7,8],[3,4,5],[0,1,2]] if numLayers==3 else [[3,4,5],[1,2,3]]
    
    boxes_wh = trueBoxes[...,2:4]*inputShape[::-1]
    trueBoxes = np.array(trueBoxes,dtype='float32')
    inputShape = np.array(inputShape,dtype='int32')
    
   # print(trueBoxes.shape)
    m = trueBoxes.shape[0]
   # print("m = ",m)
    gridShape = [inputShape//{0:32,1:16,2:8}[l] for l in range(numLayers)]
    y_true = [np.zeros((m,gridShape[l][0],gridShape[l][1],len(anchorMask[l]),5+numClasses), 
        dtype='float32') for l in range(numLayers)]

    anchors = np.expand_dims(anchors,0)
    anchorMaxes = anchors /2. 
    anchorMins = -anchors 
    validMask = boxes_wh[...,0]>0
    for b in range(m):
        wh = boxes_wh[b,validMask[b]]
        if len(wh)==0:continue 

        wh = np.expand_dims(wh,-2)
        boxMaxes = wh/2. 
        boxMins = -boxMaxes 

        intersectMins = np.maximum(boxMins,anchorMins)
        intersectMaxes = np.minimum(boxMaxes,anchorMaxes)
        intersectWh = np.maximum(intersectMaxes-intersectMins,0.) 
        intersectArea = intersectWh[...,0]*intersectWh[...,1]
        boxArea = wh[...,0]*wh[...,1]
        anchorArea = anchors[...,0]*anchors[...,1]
        iou = intersectArea/(boxArea+anchorArea-intersectArea)

        bestAnchor = np.argmax(iou,axis=-1)
        #print(trueBoxes)

        for t,n in enumerate(bestAnchor):
            for l in range(numLayers):
                if n in anchorMask[l]:
                    i = np.floor(trueBoxes[b,t,1]*gridShape[l][1]-0.0001).astype('int32')
                    j = np.floor(trueBoxes[b,t,2]*gridShape[l][0]-0.0001).astype('int32')
                    k = anchorMask[l].index(n)
                    c = trueBoxes[b,t,0].astype('int32')
       #             print("b:{},i:{},j:{},k:{},l:{}".format(b,i,j,k,l))
       #             print("current {} layer shape: {}".format(l,y_true[l].shape))
                    y_true[l][b,j,i,k,0:4] = trueBoxes[b,t,1:5]
                    y_true[l][b,j,i,k,4] = 1
                    y_true[l][b,j,i,k,5+c] = 1

    return y_true

def masking(ds,mask):
    pivot = [ classes1.index(i) for i in mask]
    maskedData = {}
    maskData ={}
    for i in ds:
        fileindex = i.split("/")[5]
        fileindex = fileindex.replace(".png",".jpg")
        filename = fileindex.replace(".jpg\n",".txt")
        imgName = fileindex.replace(".jpg\n",'.jpg')

        filepath = os.path.join("/home/Fire/labels",filename)
        imgPath = os.path.join("/home/Fire/JPEGImages",imgName)
        with open(filepath,'r') as f:
            tmp = []
            tmp2 = []
            data = f.readlines()
            flag1 = False
            flag2 = False
            for datapoint in data:
                datapoint = datapoint.replace('\n','')
                if int(datapoint.split(" ")[0]) in pivot:
                    flag2 = True
                    tmp2.append(datapoint)
                else:
                    flag1 = True
                    tmp.append(datapoint)
            if flag1: maskedData[imgPath] = tmp 
            if flag2: maskData[imgPath] = tmp2
    return maskedData,maskData

def getRandomData(imgName,bboxs,inputShape,max_boxes=20,random=True,jitter=.3,hue=.1,sat=1.5,val=1.5,proc_img=True):

#    imgName = imgName.replace('.jpg\n','.jpg')
    
    image = Image.open(imgName)
    iw,ih = image.size
#    print iw,ih
    h,w = inputShape
    h = float(h)
    w = float(w)
    bbox = np.array([np.array(list(map(float,box.split()))) for box in bboxs])

    if not random:
        scale = min(w/iw,h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        imageData = 0
        if proc_img:
            image = image.resize((nw,nh),Image.CUBIC)
            newImage = Image.new('RGB',(w,h),(128,128,128))
            newImage.paste(image,(dx,dy))
            imageData = np.array(newImage)//255. 

        
        boxData = np.zeros((max_boxes,5))
        if len(bbox)>0:
            np.random.shuffle(bbox)
            if len(bbox)>max_boxes: bbox=bbox[:max_boxes]
            bbox[:,[2,4]] = bbox[:,[2,4]]*scale
            bbox[:,[1,3]] = bbox[:,[1,3]]*scale 
            boxData[:len(bbox)] = bbox 
            
        return imageData,boxData
    
    newAr = w/h*rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25,2)
    if newAr < 1:
        nh = int(scale*h)
        nw = int(nh*newAr)
    else:
        nw = int(scale*w)
        nh = int(nw/newAr)
    image = image.resize((nw,nh),Image.CUBIC)

    dx=int(rand(0,w-nw))
    dy=int(rand(0,h-nh))
    newImage = Image.new('RGB',inputShape,(128,128,128))
    newImage.paste(image,(dx,dy))
    image = newImage 

    flip = rand()<.5
    if flip: image=image.transpose(Image.FLIP_LEFT_RIGHT)


    hue = rand(-hue,hue)
    sat = rand(1,sat) if rand() <.5 else 1/rand(1,sat)
    val = rand(1,val) if rand() <.5 else 1/rand(1,val)

    x = rgb_to_hsv(np.array(image)/255.)

    x[...,0] += hue
    x[...,0][x[...,0]>1] -=1
    x[...,0][x[...,0]<0] +=1
    x[...,1]*=sat 
    x[...,2]*=val 
    x[x>1] = 1
    x[x<0] = 0
    imageData = hsv_to_rgb(x)


    boxData = np.zeros((max_boxes,5))
    if len(bbox)>0:
        np.random.shuffle(bbox)
        bbox[:,[2,4]] = bbox[:,[2,4]]*(nh+dy)/ih 
        bbox[:,[1,3]] = bbox[:,[1,3]]*(nw+dx)/iw 
        bbox = np.clip(bbox,0,0.999999999)
        if flip: bbox[:,[1]] = 1-bbox[:,[1]]
      #  bbox[:,1:3][bbox[:,1:3]<0] = 0
        bbox[:,3][bbox[:,3]>=1] = 0.9999999 
        bbox[:,4][bbox[:,4]>=1] = 0.9999999 
#        bboxW = bbox[:,3]-bbox[:,1]
#        bboxH = bbox[:,4]-bbox[:,2]
#        bbox = bbox[np.logical_and(bboxW>1,bboxH>1)]
        if len(bbox) > max_boxes : bbox=bbox[:max_boxes]
        boxData[:len(bbox)] = bbox 
    return imageData,boxData 

def rand(a=0,b=1):
    return np.random.rand()*(b-a)+a 
    

def dataGenerator(dataset,batchSize,inputShape,anchors,numClasses):
    n = len(dataset)
    i = 0
    ds = list(dataset.items())
    while True:
        imageData = []
        boxData = []
        for b in range(batchSize):
            if i == 0:
                np.random.shuffle(ds)
            image,box = getRandomData(ds[i][0],ds[i][1],inputShape)
            imageData.append(image)
            boxData.append(box)
            i=(i+1)%n 
        #print("finish dataRandomize")
        #print("image data len",len(imageData))
        #print("box data len",len(boxData))
        #print(np.shape(boxData))
        imageData = np.array(imageData)
 #       print("img numpyfy")
        boxData = np.array(boxData)
 #       print("before preprocess true box bboxdata shape: ",boxData.shape)
        #print("finish append data and transform")
        y_true = preprocessTrueBox(boxData,inputShape,anchors,numClasses)
        #print("finish dataframe Preprocessing")
        yield [imageData, *y_true], np.zeros(batchSize)

def dataGeneratorWrapper(dataset,batchSize,inputShape,anchors,numClasses):
    n = len(dataset)
    if n == 0 or batchSize <=0: return None 
    return dataGenerator(dataset,batchSize,inputShape,anchors,numClasses)


def splitData(filePath):
    with open (filePath,'r') as f:
        data = f.readlines()
        data_sp = data
        random.shuffle(data_sp)
        #data_sp = [str(i.split('/')[5].split("_")[1].split(".")[0])+" 1\n" for i in data]
        pvt = int(len(data)/5)
        data_train = data_sp[:pvt*3]
        data_test = data_sp[pvt*3:pvt*4]
        data_val = data_sp[pvt*4:]
        #print len(data_val)
        #print data_val
        return [data_train,data_test,data_val]



def getAnchors(anchorPath):
    with open(anchorPath) as f:
        anchors = f.readlines()
    anchors = [float(x) for x in anchors[0].split(',')]
    return np.array(anchors).reshape(-1,2)






if __name__ == '__main__':
    filePath = "/home/Fire/list.txt"
    sp = splitData(filePath) 
    maskedDS,maskDS = masking(sp[0],['fire','car']) 
    ds1 = maskDS.items()
    ds2 = list(maskedDS.items())
#    np.random.shuffle(ds2)
    inputShape = (416,416)

    idata = []
    bdata = []

    anchorPath = '/home/keras-yolo3/model_data/yolo_anchors.txt'
    anchors=getAnchors(anchorPath) 
    numClasses=5

    dgw = dataGeneratorWrapper(maskDS,32,inputShape,anchors,numClasses)

    for dg in dgw:
        print(dg) 
        break

    for key,val in ds2:
        i,b = getRandomData(key,val,inputShape)
        idata.append(i)
        bdata.append(b) 
        if len(idata) > 31:break
    idata = np.array(idata)
    bdata = np.array(bdata)
    b = preprocessTrueBox(bdata,inputShape,anchors,numClasses) 
#    print(b)
    print(b)
    print(bdata.shape)
        



