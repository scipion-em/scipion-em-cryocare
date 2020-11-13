
import pyworkflow.object as pwobj
from pwem import EMObject


class CryocareTrainData(EMObject):
    def __init__(self, train_data=None, mean_std=None, patch_size=None, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._train_data = pwobj.String(train_data)
        self._mean_std = pwobj.String(mean_std)
        self._patch_size = pwobj.Integer(patch_size)

    def getTrainData(self):
        return self._train_data.get()

    def getMeanStd(self):
        return self._mean_std.get()

    def getPatchSize(self):
        return self._patch_size.get()

    def getPath(self):
        return '{}\n{}'.format(self.getTrainData(), self.getMeanStd())

    def __str__(self):
        return "CryoCARE Train Data (path=%s)" % self.getPath()


class CryocareModel(EMObject):
    def __init__(self, basedir=None, mean_std=None, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._basedir = pwobj.String(basedir)
        self._mean_std = pwobj.String(mean_std)

    def getPath(self):
        return self._basedir.get()

    def getMeanStd(self):
        return self._mean_std.get()

    def __str__(self):
        return "CryoCARE Model (path=%s)" % self.getPath()
