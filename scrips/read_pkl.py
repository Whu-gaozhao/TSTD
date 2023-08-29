import pickle



data = '/home/zhao/DataSets/scannet/change/scannet_infos_test.pkl'
with open(data, "rb") as fp_data:
    data_pkl = pickle.load(fp_data)
    print(data_pkl)
    print(len(data_pkl))