# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:47:45 2017

@author: lenovo
"""
from keras.layers import merge
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import *
from keras.layers.core import Dense,Activation,Flatten,Dropout,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input

def block_type1(x,nb_filter,filter_len=16):
    out = Conv1D(nb_filter,filter_len,padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out=Dropout(0.5)(out)
    out = Conv1D(nb_filter,filter_len,padding='same')(out)
    #out = merge([out,x],mode='sum')
    return out

def block_type2(x,nb_filter,filter_len=16):
    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out=Dropout(0.5)(out)
    out = Conv1D(nb_filter,filter_len,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out=Dropout(0.5)(out)
    out = Conv1D(nb_filter,filter_len,padding='same')(out)
    #out = merge([out,x],mode='sum')
    return out

###
#inp = Input(shape=(200,30))
inp = Input(shape=(15191,1))
inp_begin= Conv1D(64,16,padding='same')(inp)
inp_begin = BatchNormalization()(inp_begin)
inp_begin= Activation('relu')(inp_begin)
###

out_1=block_type1(inp_begin,64,filter_len=16)
#maxpooling1
inp_1=MaxPooling1D(pool_size=2,padding='valid')(inp_begin)
out_1=MaxPooling1D(pool_size=2,padding='valid')(out_1)
out_1=merge([out_1,inp_1],mode='sum')


out_2=block_type2(out_1,64,filter_len=16)

out_2=block_type2(out_2,64,filter_len=16)
#maxpooling2
out_2=MaxPooling1D(pool_size=2,padding='valid')(out_2)
inp_2=MaxPooling1D(pool_size=2,padding='valid')(inp_1)
out_2=merge([out_2,inp_2],mode='sum')

out_3=block_type2(out_2,64,filter_len=16)

inp_3= Conv1D(64*2,1,padding='same')(inp_2)

out_3=block_type2(out_3,64*2,filter_len=16)
#maxpooling3
out_3=MaxPooling1D(pool_size=2,padding='valid')(out_3)
inp_3=MaxPooling1D(pool_size=2,padding='valid')(inp_3)
out_3=merge([out_3,inp_3],mode='sum')

out_4=block_type2(out_3,64*2)

out_4=block_type2(out_4,64*2)
#maxpooling4
out_4=MaxPooling1D(pool_size=2,padding='valid')(out_4)
inp_4=MaxPooling1D(pool_size=2,padding='valid')(inp_3)
out_4=merge([out_4,inp_4],mode='sum')

out_5=block_type2(out_4,64*2)

inp_5= Conv1D(64*3,1,padding='same')(inp_4)

out_5=block_type2(out_5,64*3)
#maxpooling5
out_5=MaxPooling1D(pool_size=2,padding='valid')(out_5)
inp_5=MaxPooling1D(pool_size=2,padding='valid')(inp_5)
out_5=merge([out_5,inp_5],mode='sum')

out_6=block_type2(out_5,64*3)
out_6=block_type2(out_6,64*3)
#maxpooling 6
out_6=MaxPooling1D(pool_size=2,padding='valid')(out_6)
inp_6=MaxPooling1D(pool_size=2,padding='valid')(inp_5)
out_6=merge([out_6,inp_6],mode='sum')

out_7=block_type2(out_6,64*3)



inp_7= Conv1D(64*4,1,padding='same')(inp_6)
out_7=block_type2(out_7,64*4)
#maxpooling 7
out_7=MaxPooling1D(pool_size=2,padding='valid')(out_7)
inp_7=MaxPooling1D(pool_size=2,padding='valid')(inp_7)
out_7=merge([out_7,inp_7],mode='sum')

#out_8=block_type2(out_7,64*4)
#out_8=block_type2(out_8,64*4)
#out_8=MaxPooling1D(pool_size=2,padding='valid')(out_8)
#inp_8=MaxPooling1D(pool_size=2,padding='valid')(inp_7)
#out_8=merge([out_8,inp_8],mode='sum')
#out_8=block_type2(out_8,64*4)

out_final = BatchNormalization()(out_7)
out_final= Activation('relu')(out_final)

#out_final=Dense(14)(out_final)
out_final=Flatten()(out_final)
out_final=Dense(15191)(out_final)
out_final= Activation('softmax')(out_final)
model = Model(inp,out_final)

model.compile(optimizer='adam',loss='mse')
bspdn=np.reshape(bspdn,[54,15191,1])
model.fit(bspdn,epdn[0:54,:],epochs=50,batch_size=2)








    