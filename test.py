import pickle
from src.load_data import ProcessData
from PIL import Image
file="/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/Satellit/SatelliteImgs/model.sav"
model = pickle.load(open(file, "rb"))


img = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/Satellit/SatelliteImgs/data/nowildfire/-73.4525,45.545352.jpg"
p = ProcessData()
img = Image.open(img)

mean=[]
_, test_t=p.apply_transformations(mean, std)

imgs = test_t(img)
res=model.predict(imgs)
print(res)