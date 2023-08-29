import pickle
# f=open(r'/home/zhao/DataSets/scannet/scannet_infos_train.pkl', "rb")#文件所在路径
# inf=pickle.load(f, encoding='latin1')#读取pkl内容
# print (inf)
# f.close()

def ChangeCharacter(_in_src, _out_src, _old_char, _new_char):
    _in_open = open(_in_src, "rb")#文件所在路径
    # _out_open = open(_out_src,"wb")
    _old_pkl = pickle.load(_in_open,encoding='latin1')
    # _out_open.write(_old_pkl.replace(_old_char, _new_char))

    print(type(_old_pkl))
    _old_pkl = str(_old_pkl).replace(_old_char, _new_char)
    _old_pkl = list(_old_pkl)
    with open(_out_src, "wb") as fp_data:
        pickle.dump(_old_pkl,fp_data)
    _in_open.close()
    fp_data.close()


if __name__ == '__main__':
    in_src = '/home/zhao/DataSets/scannet/scannet_infos_train.pkl'
    out_src = '/home/zhao/DataSets/scannet/scannet_infos_train_new.pkl'
    ChangeCharacter(_in_src=in_src, _out_src=out_src, _old_char='\\', _new_char='/')

