
import pyworkflow.object as pwobj
from pwem import EMObject


class CryocareTrainData(EMObject):
    def __init__(self, train_data_dir=None, patch_size=None, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._train_data_dir = pwobj.String(train_data_dir)
        self._patch_size = pwobj.Integer(patch_size)

    def getTrainDataDir(self):
        return self._train_data_dir.get()

    def getPatchSize(self):
        return self._patch_size.get()

    def __str__(self):
        return "CryoCARE Train Data (path=%s)" % self.getTrainDataDir()


class CryocareModel(EMObject):
    def __init__(self, basedir=None, train_data_dir=None, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._basedir = pwobj.String(basedir)
        self._train_data_dir = pwobj.String(train_data_dir)

    def getPath(self):
        return self._basedir.get()

    def getTrainDataDir(self):
        return self._train_data_dir.get()

    def __str__(self):
        return "CryoCARE Model (path=%s)" % self.getPath()
