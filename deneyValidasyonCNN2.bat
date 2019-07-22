rem python extractFeature.py -f CNN -exp deneyValidasyonCNN2 -w D:\Apps\ActionDeney\CNNTrainValidasyonsuz\weights.hdf5
python hmmTrain.py -s 3 -l2r -exp deneyValidasyonCNN2 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deneyValidasyonCNN2 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22