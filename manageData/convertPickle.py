import pickle
import numpy
#with open("traffic-signs-data/train.p", "rb") as f:
with open("traffic-signs-data/test.p", "rb") as f:
    w = pickle.load(f)

#pickle.dump(w, open("traffic-signs-data/trainData.p","wb"), protocol=2)
pickle.dump(w, open("traffic-signs-data/testData.p","wb"), protocol=2)