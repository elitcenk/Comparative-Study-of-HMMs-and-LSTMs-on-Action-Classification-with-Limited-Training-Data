REM python extractFeature.py -f CNN -exp deneyNewCNN5 -w D:\Apps\ActionDeney\trainCNN\weights.hdf5
python hmmTrain.py -s 7 -exp deneyNewCNN5 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deneyNewCNN5 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22