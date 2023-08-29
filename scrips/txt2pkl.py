import pickle
# Read as bytes
with open('scannet_infos_train.txt', 'rb') as f:
    data =  f.read()

# Save as pickle
with open('scannet_infos_train.pkl', 'wb') as f:
    pickle.dump(data, f)

# # Load as pickle
# with open('s3dis_infos_Area_1.pkl', 'rb') as f:
#    data_pickle = pickle.load(f)