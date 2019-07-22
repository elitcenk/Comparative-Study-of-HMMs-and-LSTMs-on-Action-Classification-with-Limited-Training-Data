rem python extractFeature.py -f CNN -exp deneyValidasyonCNN2 -w D:\Apps\ActionDeney\CNNTrainValidasyonsuz\weights.hdf5
python hmmTrain.py -s 3 -exp deneyValidasyonCNN -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deneyValidasyonCNN -tp person02 person03 person05 person06 person07 person08 person09 person10 person22