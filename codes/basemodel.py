import sys, os, math, pickle, time
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
#from sklearn.cross_validation import train_test_split
from keras.models import Sequential,Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Input, concatenate,ZeroPadding2D
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras.applications import inception_v3
from keras.applications import densenet


BATCH_SIZE = 32
MAX_EPOCH  = 500 #1000 #50
MODELDIR   = '../model/%s.hdf5'
PREDDIR    = '../prediction/%s.txt'

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=16, dim=(50,1000), n_channels=9,#14,
                 n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
 
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        '''
        X_keys = ['tumor_raw_ref', 'tumor_raw_alt',  'tumor_processed_ref', 'tumor_processed_alt', 
                  'normal_raw_ref', 'normal_raw_alt','normal_processed_ref', 'normal_processed_alt', 
                  'lodt', 'tri_context', 'lodn']
        y_key  = 'label'
        '''
        #X_keys = ['read_image', 'vaf_image']
        X_keys = ['read_image', 'vaf_hist_image']
        #X_keys = ['read_image', 'vaf_list']
        y_key = 'true_ratio'
        vaf_dim = (100, 101)
        X = {}
        
        #X[X_key] = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        for key in X_keys:
            if key in ['vaf-list']:
                #X[key] = np.empty((self.batch_size, *vaf_dim, 1))
                X[key] = []
            elif key in ['vaf_hist_image']:
                X[key] = np.empty((self.batch_size, *vaf_dim, 1))
            else:
                X[key] = np.empty((self.batch_size, *self.dim, self.n_channels))
                #X['normal_contam'] = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        y = np.empty((self.batch_size,self.n_classes), dtype=float)
        for j in range(5) :
            X['vaf'+str(j+1)] = np.empty(self.batch_size)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #data = pickle.load(open('../data/%s.pkl' %ID,'rb'))
            data = pickle.load(open('%s' %ID, 'rb'))
            #contam = int(ID.split('/')[8][1:3]) / 100.0
            #contam = int(ID.split('/')[6][22:24]) / 100.0
            #X[X_key][i,] = data
            #X[X_key][i,] = np.delete(data, [4,5,6,7], axis=2)
            
            for key in X_keys:
                #if type(data[key]) == np.ndarray:
                #    X[key][i,] = np.swapaxes(data[key].T,0,1)
                #else:
                #    X[key][i,] = data[key]
                #X[key][i,] = data[key]
                #print(data[key].shape)
                #print(data[key][:,:,:])

                #if key in ['lodt', 'tri_context', 'lodn']:
                #    X[key][i,] = data[key]
                #else:
                #    X[key][i,] = np.delete(data[key], [11], axis=2)
                if key in ['vaf_list'] :
                    #X[key][i,] = np.delete(np.reshape(data[key], data[key].shape + (1,)), range(20, 100), axis=0)
                    for j in range(5) :
                        #X['vaf' + str(j+1)] = np.empty(self.batch_size)
                        X['vaf' + str(j+1)][i,] = data[key][j]
                        #print(data[key][j])
                #if key in ['vaf_hist_image'] :
                #    X[key][i,] = data[key][:,:,np.newaxis]
                else :
                    X[key][i,] = data[key]
                    #X[key][i,] = np.delete(data[key], range(500, 1000), axis=1)
                    #X['normal_contam'][i,] = np.delete(data[key], range(100, 1000), axis=1)
            

            y[i,] = data[y_key]
            #y[i,] = int(ID.split('/')[7].split('_')[0].split('n')[1].split('t')[0]) / 100.0
        return X, y

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def small_model(): #081318
    img_shape=(50,101,14)
    
    input_tumor_raw_ref = Input(shape=img_shape,name='tumor_raw_ref')
    input_tumor_raw_alt = Input(shape=img_shape,name='tumor_raw_alt')
    input_tumor_processed_ref = Input(shape=img_shape,name='tumor_processed_ref')
    input_tumor_processed_alt = Input(shape=img_shape,name='tumor_processed_alt')
    
    input_normal_raw_ref = Input(shape=img_shape,name='normal_raw_ref')
    input_normal_raw_alt = Input(shape=img_shape,name='normal_raw_alt')
    input_normal_processed_ref = Input(shape=img_shape,name='normal_processed_ref')
    input_normal_processed_alt = Input(shape=img_shape,name='normal_processed_alt')
    
    input_lodt=Input(shape=(1,),name='lodt')
    input_lodn=Input(shape=(1,),name='lodn')
    input_tri_context=Input(shape=(1,),name='tri_context')
    
    to_merge = [input_lodt,input_lodn,input_tri_context]
    
    
    img_list = [[input_tumor_raw_ref,'tumor_raw_ref'],
                [input_tumor_raw_alt,'tumor_raw_alt'],
                [input_tumor_processed_ref,'tumor_processed_ref'],
                [input_tumor_processed_alt,'tumor_processed_alt'],
                [input_normal_raw_ref,'normal_raw_ref'],
                [input_normal_raw_alt,'normal_raw_alt'],
                [input_normal_processed_ref,'normal_processed_ref'],
                [input_normal_processed_alt,'normal_processed_alt']]
    for input_img,imgkeys in img_list:
        
        conv_model = model_lite(input_tensor=input_img,input_shape=img_shape)
        for layer in conv_model.layers[1:]:
            layer.name = layer.name+imgkeys
        x = conv_model.output
        #x = Flatten()(x)
        to_merge.append(x)
        
    merged = concatenate(to_merge)
    fully  = Dense(64,kernel_initializer='he_uniform',activation='relu')(merged)
    preds  = Dense(1, activation='sigmoid',name='output')(fully)
    model = Model(inputs=[input_tumor_raw_ref,       input_tumor_raw_alt,
                          input_tumor_processed_ref, input_tumor_processed_alt,
                          input_normal_raw_ref,      input_normal_raw_alt,
                          input_normal_processed_ref,input_normal_processed_alt,
                          input_lodt,input_lodn,input_tri_context],
                  outputs=preds)
    
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy',precision,recall])
    print(model.summary())
    return model

def small_model_for_test(): #081318
    #img_shape=(50,101,14)
    #img_shape=(50,1000,9)
    img_shape=(50,1000,9)
    vaf_shape=(100,101,1)
    #input_tumor_raw_ref = Input(shape=img_shape,name='tumor_raw_ref')
    #input_tumor_raw_alt = Input(shape=img_shape,name='tumor_raw_alt')
    #input_tumor_processed_ref = Input(shape=img_shape,name='tumor_processed_ref')
    #input_tumor_processed_alt = Input(shape=img_shape,name='tumor_processed_alt')

    #input_normal_raw_ref = Input(shape=img_shape,name='normal_raw_ref')
    #input_normal_raw_alt = Input(shape=img_shape,name='normal_raw_alt')
    #input_normal_processed_ref = Input(shape=img_shape,name='normal_processed_ref')
    #input_normal_processed_alt = Input(shape=img_shape,name='normal_processed_alt')

    #input_lodt=Input(shape=(1,),name='lodt')
    #input_lodn=Input(shape=(1,),name='lodn')
    #input_tri_context=Input(shape=(1,),name='tri_context')

    #to_merge = [input_lodt,input_lodn,input_tri_context]
    #to_merge = [input_lodt, input_lodn, input_tri_context]
    #to_merge = []
    input_image = Input(shape=img_shape, name = 'read_image')
    #input_image = Input(shape=vaf_shape, name = 'vaf_hist_image')
    input_vaf_hist = Input(shape=vaf_shape, name = 'vaf_hist_image')
    
    input_vaf1 = Input(shape=(1,), name = 'vaf1')
    input_vaf2 = Input(shape=(1,), name = 'vaf2')
    input_vaf3 = Input(shape=(1,), name = 'vaf3')
    input_vaf4 = Input(shape=(1,), name = 'vaf4')
    input_vaf5 = Input(shape=(1,), name = 'vaf5')

    #for i in range(5) :
    #    input_vaf_each = Input(shape=(1,) name='vaf'+str(i))
    #    input_vaf[i] = input_vaf_each
    #    to_merge.append(input_vaf_each)
    #input_vaf = Input(shape=vaf_shape, name = 'vaf_image')
    #to_merge = [input_vaf1, input_vaf2, input_vaf3, input_vaf4, input_vaf5]
    #to_merge = [input_vaf1]
    to_merge = []
    '''
    img_list = [[input_tumor_raw_ref,'tumor_raw_ref'],
                [input_tumor_raw_alt,'tumor_raw_alt'],
                [input_tumor_processed_ref,'tumor_processed_ref'],
                [input_tumor_processed_alt,'tumor_processed_alt'],
                [input_normal_raw_ref,'normal_raw_ref'],
                [input_normal_raw_alt,'normal_raw_alt'],
                [input_normal_processed_ref,'normal_processed_ref'],
                [input_normal_processed_alt,'normal_processed_alt']]
    '''
    img_list = [[input_image, 'read_image', img_shape], [input_vaf_hist, 'vaf_hist_image', vaf_shape]]

    conv_model_output = ''
    for input_img,imgkeys,shape in img_list:
        conv_model = model_lite(input_tensor=input_img,input_shape=shape, img_keys = imgkeys)

        for layer in conv_model.layers[1:]:
            layer.name = layer.name+imgkeys
        #if 'read_image' in imgkeys :
        #    conv_out = Dense(128,kernel_initializer='he_uniform',activation='relu',name='fc_1')(conv)
        x = conv_model.output
        conv_model_output = conv_model.output
        #x = Flatten()(x)
        to_merge.append(x)

    '''
    img_list = [[input_vaf, 'vaf_image']]
    for input_img, imgkeys in img_list:
        conv_model = model_lite(input_tensor=input_img, input_shape=vaf_shape)

        for layer in conv_model.layers[1:]:
            layer.name = layer.name+imgkeys
        x = conv_model.output
        to_merge.append(x)
    '''
    #for input_img,imgkeys in img_list:
    #input_contam = Input(shape=img_shape, name = 'normal_contam')
    #img_list = [[input_contam, 'normal_contam']]
    
    #conv_model = model_lite(input_tensor=input_contam,input_shape=img_shape)
    
    '''    
    for layer in conv_model.layers[1:]:
            layer.name = layer.name+imgkeys
        x = conv_model.output
        #x = Flatten()(x)
        to_merge.append(x)
    '''
    merged = concatenate(to_merge)
    #merged = concatenate(to_merge)
    fully  = Dense(16,kernel_initializer='he_uniform',activation='relu')(merged)
    #fully  = Dense(2,kernel_initializer='he_uniform',activation='relu')(merged)
    preds  = Dense(1, activation=None, name='output')(fully)#merged)#conv_model_output)#fully)#(merged)#(conv_model.output) #(fully)

    #model = Model(inputs=[input_image, input_vaf],
    #              outputs=preds)

    model = Model(inputs=[input_image, input_vaf_hist], outputs=preds)

    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy',precision,recall])
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def model_lite(input_tensor,input_shape, img_keys):
    chanDim = -1
    conv = Conv2D(8, (2, 1), padding="valid", 
                  input_shape=input_shape,name='conv_1')(input_tensor)
    #conv = Activation("relu",name='relu_1')(Conv2D(16, (3, 1), padding="same",name='conv_3')(conv))
    conv = Conv2D(16, (3, 1), padding="same", name = 'conv_3')(conv)
    #conv = Activation("relu",name='relu_3')(conv)
    #conv = BatchNormalization(axis=chanDim,name='BN_3')(conv)
     #conv = AveragePooling2D(pool_size=(2, 2),name='pool_2')(conv)
    #conv = MaxPooling2D(pool_size=(2,2), name='pool_2')(conv)
    conv = Dropout(0.3,name='DO_2')(conv)

    #conv = Conv2D(64, (1, 3), padding="same",name='conv_4')(conv)
    #conv = Activation("relu",name='relu_4')(conv)
    #conv = BatchNormalization(axis=chanDim,name='BN_4')(conv)
    #conv = Conv2D(128, (3, 3), padding="same",name='conv_5')(conv)
    #conv = Activation("relu",name='relu_5')(conv)
    #conv = BatchNormalization(axis=chanDim,name='BN_5')(conv)
    #conv = AveragePooling2D(pool_size=(2, 2),name='pool_3')(conv)
    #conv = Dropout(0.2,name='DO_3')(conv)
    
    conv = Flatten(name='flatten')(conv)
    #conv_out = Flatten(name='flatten')(conv)

    conv_out = Dense(128,kernel_initializer='he_uniform',activation='relu',name='fc_1')(conv)
    #conv_out = Dropout(0.2,name='DO_4')(conv_out)
    #conv_out = Dense(32,kernel_initializer='he_uniform',activation='relu',name='fc_2')(conv)
    #conv_out = Dense(1, activation=None, name='fc_2')(conv_out)
    if img_keys == 'read_image' :
        conv_out = Dense(3, kernel_initializer='he_uniform',activation='relu',name='fc_read')(conv_out)

    model= Model(inputs=[input_tensor], outputs=conv_out)
    return model


def example_data():
    data_1 = pickle.load(open('../data/data_example1.pkl','rb'))
    keylist = list(data_1.keys())
    dat_dict = {key:[] for key in keylist}
    #print(keylist)
    for datid in range(1,1+128):
        data = pickle.load(open('../data/data_example%d.pkl' %datid,'rb'))
        for key in keylist:
            if type(data[key]) == np.ndarray:
                dat_dict[key].append(np.swapaxes(data[key].T,0,1))
            elif type(data[key]) == str:
                dat_dict[key].append({'gn':[0,1],'gp':[1,0]}[data[key]])
            else:
                dat_dict[key].append(data[key])
    for key in dat_dict:
        dat_dict[key] = np.array(dat_dict[key])
        #print(key,dat_dict[key].shape)
    return dat_dict

def main_learner():
    data  = example_data()
    model = baseline_model()
    train_x= {key:data[key] for key in data if key != 'label'}
    train_y= data['label']
    print('Model loaded, start training', time.ctime())
    history=model.fit(x=train_x,y=train_y,batch_size=BATCH_SIZE,epochs=MAX_EPOCH,
                     verbose=True)
    return None

def store_baseline_model(model_name):
    model = baseline_model()
    model.save( MODELDIR %model_name)
    
def store_densenet_model(model_name):
    model = denesenet_model()
    model.save(MODELDIR %model_name)

def store_small_model(model_name):
    model = small_model()
    model.save(MODELDIR %model_name)

def store_small_model_only_raw(model_name):
    model = small_model_only_raw()
    model.save(MODELDIR % model_name)

def store_small_model_only_processed(model_name):
    model = small_model_only_processed()
    model.save(MODELDIR % model_name)

def store_small_model_for_test(model_name):
    model = small_model_for_test()
    model.save(MODELDIR % model_name)

def main_learner_fgen(trainingset='file_train_gp_oversampled',
                      model_name='base_initial',out_name='baseline_trained'):
    print('Start', time.ctime())
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model = load_model(MODELDIR %model_name,custom_objects={'precision':precision,
                                                        'recall':recall})
    print('Model loaded, start training', time.ctime())
    IDlist = [i.strip() for i in open('../ref/%s.txt' %trainingset)]#[:64]####
    size = len(IDlist)
    test_size = int(size * 0.2)
    train_size = size - test_size
    IDlist_random = np.random.choice(IDlist, size, replace = False)
    IDlist_train = IDlist_random[:train_size]
    IDlist_test = IDlist_random[train_size:]

    steps_per_epoch = np.ceil(len(IDlist_train)/BATCH_SIZE)
    params = {'dim': (50,1000),
          'batch_size': BATCH_SIZE,
          'n_classes': 1,
          'n_channels': 9, #14, 
          'shuffle': True}

    generator_train = DataGenerator(IDlist_train,**params)
    generator_test = DataGenerator(IDlist_test,**params)

    #print (generator[0][0]) 
    #X = generator[0]
    #y = generator[1]

    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    #train = (X_train, y_train)
    #test = (X_test, y_test)

    model_ckpt   = ModelCheckpoint(MODELDIR %out_name, monitor='loss',save_best_only=True,mode='min')

    #history=model.fit_generator(generator,steps_per_epoch=steps_per_epoch,
    #                            epochs=MAX_EPOCH,callbacks=[model_ckpt],
    #                            verbose=1)
    history=model.fit_generator(generator_train,steps_per_epoch=steps_per_epoch,
                                epochs=MAX_EPOCH,callbacks=[model_ckpt], validation_data = generator_test,
                                verbose=1)

    print('Ended training',time.ctime())

def base_prediction(testset = 'file_testset',model_name = 'baseline_trained',
                   out_file = 'base_pred'):
    print('Start', time.ctime())
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model = load_model(MODELDIR %model_name,custom_objects={'precision':precision,
                                                        'recall':recall})

    ###
    BATCH_SIZE = 1
    ###
    print('Model loaded, start prediction', time.ctime())
    
    IDlist = [i.strip() for i in open('../ref/%s.txt' %testset)]
    params = {'dim': (50,1000),
          'batch_size': BATCH_SIZE,
          'n_classes': 1,
          'n_channels': 9, #14,
          'shuffle': False}
    generator = DataGenerator(IDlist,**params)
    
    steps = np.ceil(len(IDlist)/BATCH_SIZE)
    #print("!")
    preds = model.predict_generator(generator,steps=steps)
    #print("!!")
    #y_labels= [pickle.load(open('../data/%s.pkl' %ID,'rb'))['label'] for ID in IDlist]
    #y_labels= [pickle.load(open('%s' %ID,'rb'))['label'] for ID in IDlist]
    #y_labels= [pickle.load(open('%s' %ID,'rb'))['label'] for ID in IDlist]
    #y_labels = [int(ID.split('/')[8][1:3]) / 100.0 for ID in IDlist]
    #y_labels = [int(ID.split('/')[6][22:24]) / 100.0 for ID in IDlist]
    y_labels = [pickle.load(open('%s' %ID, 'rb'))['true_ratio'] for ID in IDlist]
    #y_labels = [int(ID.split('/')[7].split('_')[0].split('n')[1].split('t')[0]) / 100.0 for ID in IDlist]
    print(preds.shape)
    with open(PREDDIR %out_file,'w') as OutF:
        for ID,y_label,pred in zip(IDlist,y_labels,preds):
            print(*[ID,y_label,*pred],sep='\t',file = OutF)

def get_weights(model_name, out_file):
    print('Start', time.ctime())
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model = load_model(MODELDIR %model_name,custom_objects={'precision':precision,
                                                        'recall':recall})
    layer_list = ['lodt', 'lodn', 'tri_context']
    for each in ['tumor_raw_ref', 'tumor_raw_alt', 'tumor_processed_ref', 'tumor_processed_alt',
                'normal_raw_ref', 'normal_raw_alt', 'normal_processed_ref', 'normal_processed_alt'] :
        for i in range(16) :
            layer_list.append(each)

    with open('../weights/%s.txt' % out_file, 'w') as OutF:
        d1 = model.get_layer(name = "dense_1").get_weights()
        for idx, each in enumerate(d1[0]) :
            print(layer_list[idx], each, sep = '\t', file = OutF)
        #for layer in model.layers:
            #print(layer.name, file = OutF)
            #print(layer[0].w.get_value(), file = OutF)
            #for each in layer.get_weights() :
                #print(each.shape, each, sep='\t', file = OutF)

    #print(model.layers[0].w.get_value(), file = OutF)
    
#main_learner()
#store_baseline_model()
#main_learner_fgen()
#base_prediction()
MODE = sys.argv[1]
PARAM= sys.argv[2:]
if __name__ == '__main__':
    if MODE in locals().keys(): locals()[MODE](*PARAM)
    else: sys.exit('error: cmd=%s' %(MODE))
