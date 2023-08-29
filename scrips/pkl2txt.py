import pickle
f=open(r'/home/zhao/DataSets/scannet/scannet_infos_train.pkl', "rb")#文件所在路径
inf=pickle.load(f, encoding='latin1')#读取pkl内容
print (inf)
f.close()


inf=str(inf)
ft = open('scannet_infos_train.txt', 'w')  
ft.write(inf) 
