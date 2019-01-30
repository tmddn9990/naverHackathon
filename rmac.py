# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import scipy.io
import numpy as np

import nsml
from nsml import DATASET_PATH

import keras
from keras.models import Model, Sequential
from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.preprocessing import image
import keras.backend as K

from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

K.set_image_dim_ordering('th')

import warnings
warnings.filterwarnings("ignore")




def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def multi_input(generator, regions):
    while True:
        X1 = next(generator)
        X2 = np.expand_dims(regions, axis=0)
        yield [X1, X2]




def rmac(model, num_rois):
    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')
    # ROI pooling
    x = RoiPooling([1], num_rois)([model.layers[-3].output, in_roi])
    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)
    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)
    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)
    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)
    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)
    
    # Define model
    model = Model(input=[model.input, in_roi], output=rmac_norm)
    ## Load PCA weights
    #mat = scipy.io.loadmat('data/PCAmatrices.mat')
    #b = np.squeeze(mat['bias'], axis=1)
    #w = np.transpose(mat['weights'])
    #model.layers[-4].set_weights([w, b])
    return model


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm




def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')



    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        references = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        references = [v.split('/')[-1].split('.')[0] for v in references]
        queries.sort()
        references.sort()

        # Load RMAC model
        Wmap, Hmap = get_size_vgg_feat_map(224, 224)
        regions = rmac_regions(Wmap, Hmap, 3)
        rmac_model = rmac(model, len(regions))
        rmac_model.summary()


        test_datagen = ImageDataGenerator(dtype='float32')
        query_generator = test_datagen.flow_from_directory(
            directory=test_path,
            target_size=(224, 224),
            classes=['query'],
            color_mode="rgb",
            batch_size=1,
            class_mode=None,
            shuffle=False
        )
        query_vecs = rmac_model.predict_generator(multi_input(query_generator, regions), steps=len(query_generator), verbose=1)



        reference_generator = test_datagen.flow_from_directory(
            directory=test_path,
            target_size=(224, 224),
            classes=['reference'],
            color_mode="rgb",
            batch_size=1,
            class_mode=None,
            shuffle=False
        )
        reference_vecs = rmac_model.predict_generator(multi_input(reference_generator, regions), steps=len(reference_generator), verbose=1)
    
        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)
        
        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))
    
    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)




if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=500)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()



    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (3, 224, 224)  # input image shape


    """ Model """
    resnet_model = ResNet50(weights='imagenet', input_shape=(3,224,224))
    out = Dense(num_classes, activation='softmax')(resnet_model.layers[-2].output)
    model = Model(inputs=vgg16_model.input, output=out)
    model.summary()

    bind_model(model)
    
    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.Adam(lr=0.000045, decay=1e-6)
        model = multi_gpu_model(model, gpus=2)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])


        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.1)

        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=(224,224),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset='training'
        )
        val_generator= train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=(224,224),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset='validation'
        )

        """ Callback """
        monitor = 'val_acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)


        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      validation_data=val_generator,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True,
                                      validation_steps=val_generator.n // batch_size)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            val_loss, val_acc = res.history['val_loss'][0], res.history['val_acc'][0]
            nsml.report(summary=True, step=epoch, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))