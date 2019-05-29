import pandas as pd
from matplotlib.pyplot import imread
import glob
from mxnet import gluon
import numpy as np
import os

from mxnet.gluon.data import dataset
import warnings
from mxnet import image


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def create_class_folder_inside_dir(dir, num_class=103):
    for i in range(num_class):
        class_dir=os.path.join(dir,str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def get_string_from_file(file_path):
    text = [line.rstrip('\n') for line in open(file_path)]
    return text

class ImageFolderDatasetCustomized(dataset.Dataset):
    """A dataset for loading image files stored in a folder structure.

    like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable, default None
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
    """
    def __init__(self, root, flag=1, transform=None, sub_class_inside=True):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._sub_class = sub_class_inside
        self._list_images(self._root)

    def _list_images(self, root):
        self.synsets = []
        self.items = []

        if(self._sub_class):
            class_dirs = os.listdir(root)
            try:
                class_dirs.sort(key=int)
            except ValueError:
                class_dirs = sorted(os.listdir(root))
                pass

            for folder in class_dirs:
                path = os.path.join(root, folder)
                if not os.path.isdir(path):
                    warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                    continue
                label = len(self.synsets)
                self.synsets.append(folder)
                for filename in sorted(os.listdir(path)):
                    filename = os.path.join(path, filename)
                    ext = os.path.splitext(filename)[1]
                    if ext.lower() not in self._exts:
                        warnings.warn('Ignoring %s of type %s. Only support %s' % (
                            filename, ext, ', '.join(self._exts)))
                        continue
                    self.items.append((filename, label))
        else:
            for filename in sorted(os.listdir(root)):
                raw_name=(os.path.splitext(filename)[0]).split('_')
                length=len(raw_name)
                name_wo_ext = int(raw_name[length-1])
                filename = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s' % (
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, name_wo_ext))


    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        if(self._sub_class):
            name = os.path.basename(self.items[idx][0])
            raw_name = (os.path.splitext(name)[0]).split('_')
            length = len(raw_name)
            name_wo_ext = int(raw_name[length - 1])
            return img, label, name_wo_ext
        else:
            return img, self.items[idx][1]

    def __len__(self):
        return len(self.items)
