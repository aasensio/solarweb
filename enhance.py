import numpy as np
import os
import time

# To deactivate warnings: https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import models as nn_model

class enhance(object):

    def __init__(self):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

    def define_network(self, image, network='intensity'):

        self.image = image
        self.nx = image.shape[1]
        self.ny = image.shape[0]

        self.model = nn_model.keepsize(self.ny, self.nx, 0.0, 5, n_filters=64, l2_reg=1e-7)
        
        #print("Loading weights...")
        if (network == 'intensity'):
            self.model.load_weights("network/intensity_weights.hdf5")

        if (network == 'magnetogram'):
            self.model.load_weights("network/blos_weights.hdf5")
    
    def predict(self):
        #print("Predicting validation data...")

        input_validation = np.zeros((1,self.ny,self.nx,1), dtype='float32')
        input_validation[0,:,:,0] = self.image
        
        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        #print("Prediction took {0:3.2} seconds...".format(end-start))

        return out[0,:,:,0], end-start
            
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-i','--input', help='input')
    parser.add_argument('-o','--out', help='out')
    parser.add_argument('-d','--depth', help='depth', default=5)
    parser.add_argument('-m','--model', help='model', choices=['encdec', 'encdec_reflect', 'keepsize_zero', 'keepsize'], default='keepsize')
    parser.add_argument('-c','--activation', help='Activation', choices=['relu', 'elu'], default='relu')
    # parser.add_argument('-a','--action', help='action', choices=['cube', 'movie'], default='cube')
    parser.add_argument('-t','--type', help='type', choices=['intensity', 'blos'], default='intensity')
    parsed = vars(parser.parse_args())

    f = fits.open(parsed['input'])
    imgs = f[0].data

    print('Model : {0}'.format(parsed['type']))
    out = enhance('{0}'.format(parsed['input']), depth=int(parsed['depth']), model=parsed['model'], activation=parsed['activation'],ntype=parsed['type'], output=parsed['out'])
    out.define_network(image=imgs)
    out.predict()
    # To avoid the TF_DeleteStatus message:
    # https://github.com/tensorflow/tensorflow/issues/3388
    ktf.clear_session()
    
    # python enhance.py -i samples/hmi.fits -t intensity -o output/hmi_enhanced.fits

    # python enhance.py -i samples/blos.fits -t blos -o output/blos_enhanced.fits
