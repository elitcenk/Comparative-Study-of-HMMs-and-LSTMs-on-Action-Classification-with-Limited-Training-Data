REM python extractFeature.py -f CNN -exp deneyNewCNN -w D:\Apps\ActionDeney\trainCNN\weights.hdf5
python hmmTrain.py -s 3 -exp deneyNewCNN -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deneyNewCNN -tp person02 person03 person05 person06 person07 person08 person09 person10 person22