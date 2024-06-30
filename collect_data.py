import pandas
import glob, os
import pickle

data_lying=[]

for file_name in glob.glob("data_radiology_*"):
    data=pandas.read_pickle(file_name, compression='infer', storage_options=None)
    for i in range(len(data)):
        data_lying.append(data[i].split("|")[-1][2:])

print(data_lying, len(data_lying))

with open("lying_radiology_all", "wb") as fp:   #Pickling
    pickle.dump(data_lying, fp) 


