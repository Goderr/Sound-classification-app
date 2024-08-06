import pandas as pd
import numpy as np
from model import model
from Feature_extraction import features_extractor
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('train/train.csv')
extracted_features=[]
for index_num,row in tqdm(df.iterrows()):
    file_name = os.path.join('train/Train', str(row.ID) + '.wav')
    final_class_labels=row["Class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.to_pickle("extracted_df.pkl")


X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test),  verbose=1)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1]*100)

joblib.dump(model, 'model.joblib')

joblib.dump(labelencoder, 'label_encoder.joblib')