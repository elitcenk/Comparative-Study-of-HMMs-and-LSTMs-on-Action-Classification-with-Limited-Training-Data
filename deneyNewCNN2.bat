python extractFeature.py -f CNN -exp deneyNewCNN2 -w D:\Apps\ActionDeney\trainCNN\weights.hdf5
python hmmTrain.py -s 3 -l2r -exp deneyNewCNN2 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deneyNewCNN2 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22