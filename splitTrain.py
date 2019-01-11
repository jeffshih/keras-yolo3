"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K 
import tensorflow as tf 
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import yolo_body, tiny_yolo_body, yolo_loss
from yolo3.iyutils import splitData, masking, dataGeneratorWrapper






def _main():

    annotation_path = '/home/Fire/list.txt'
    log_dir = 'logs/000/'
    classes_path = '/home/Fire/class.name'
    anchors_path = 'model_data/yolo_anchors.txt'
    mask = ['fire','car']
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    print("get anchors")
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw
    print("length of anchors is {}".format(len(anchors)))
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    print("prepare dataset")

    sp = splitData(annotation_path) 
    trainDS = sp[0]
    valDS = sp[1]
    train_first_DS,train_second_DS=masking(trainDS,mask)
    val_first_DS,val_second_DS=masking(valDS,mask)



    np.random.seed(10101)
    numVal1st = len(val_first_DS)
    numVal2nd = len(val_second_DS)
    numTrain1st = len(train_first_DS)
    numTrain2nd = len(train_second_DS)

    print("train with trainset 1, datapoints:{}, validate with valset1, datapoints:{}".format(numTrain1st,numVal1st))

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 8
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(numTrain1st, numVal1st, batch_size))
        model.fit_generator(dataGeneratorWrapper(train_first_DS, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, numTrain1st//batch_size),
                validation_data=dataGeneratorWrapper(val_first_DS, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, numVal1st//batch_size),
                epochs=1,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_test.h5')
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')
        batch_size = 8
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(numTrain1st, numVal1st, batch_size))
        model.fit_generator(dataGeneratorWrapper(train_first_DS, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, numTrain1st//batch_size),
            validation_data=dataGeneratorWrapper(val_first_DS, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, numVal1st//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint])
 
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4 
    K.set_session(tf.Session(config=config))

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


if __name__ == '__main__':
    _main()
