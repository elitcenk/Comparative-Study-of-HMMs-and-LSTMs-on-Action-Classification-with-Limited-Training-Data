REM python extractFeature.py -f CNN -exp deneyNewCNN9 -w D:\Apps\ActionDeney\trainCNN\weights.hdf5
python hmmTrain.py -s 9 -exp deneyNewCNN9 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deneyNewCNN9 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22