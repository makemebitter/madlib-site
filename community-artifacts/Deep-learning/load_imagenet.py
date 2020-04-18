import sys
sys.path.append('/home/gpadmin/.local/lib/python3.5/site-packages/')
sys.path.remove('/usr/local/gpdb/lib/python')
print(sys.path)
import h5py
import numpy as np
import psycopg2
import glob
import os
from keras.datasets import cifar10

from madlib_image_loader import ImageLoader, DbCredentials


class ImageNetLoader(object):
    def __init__(self, db_creds, num_workers=1):
        self.connection = psycopg2.connect(user=db_creds.user,
                                           password=db_creds.password,
                                           host=db_creds.host,
                                           port=db_creds.port,
                                           database=db_creds.db_name)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        self.iloader = ImageLoader(num_workers=num_workers, db_creds=db_creds)
    def drop_table(self, name):
        self.cursor.execute("DROP TABLE IF EXISTS {}".format(name)) 
    def load_one(self, file_path, name, force=False):
        print ("Loading {}".format(file_path))
        exists = self.if_exists_table(name)
        if exists and not force:
            raise Exception("Table {} already exists!".format(name))
        h5f = h5py.File(file_path, 'r')
        np_images = np.asarray(h5f.get("images"))
        np_labels = np.eye(1000)[np.asarray(h5f.get("labels")).astype(int)]
        
        self.iloader.load_dataset_from_np(np_images, np_labels, name, append=exists)
    def load_many(self, file_list, name, force=False):
        exists = self.if_exists_table(name)
        if exists and not force:
            raise Exception("Table {} already exists!".format(name))
        for file_path in file_list:
            self.load_one(file_path, name, True)
    def if_exists_table(self, name):
        res = None
        try:
            self.cursor.execute("SELECT '{}'::regclass".format(name))
            res = self.cursor.fetchone()
        except Exception:
            pass
        return res is not None

def get_all_h5(fdir):
    return sorted(glob.glob(os.path.join(fdir, '*.h5')))
if __name__ == "__main__":
    
    db_creds = DbCredentials(db_name='cerebro',
        user='gpadmin',
                              host='localhost',
                              port='5432',
                              password='')
    imagenet_loader = ImageNetLoader(db_creds, 16)
    train_root = '/mnt/imagenet/train'
    valid_root = '/mnt/imagenet/valid'
    name_list = ['imagenet_train_data', 'imagenet_valid_data']
    file_list_list = [get_all_h5(train_root), get_all_h5(valid_root)]
    for name, file_list in zip(name_list, file_list_list):
        imagenet_loader.drop_table(name)
        imagenet_loader.load_many(file_list, name)