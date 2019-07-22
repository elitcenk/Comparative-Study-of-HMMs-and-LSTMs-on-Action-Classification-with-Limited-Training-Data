python extractFeature.py -mhi -f Hu -exp deney2
python hmmTrain.py -s 3 -l2r -exp deney2 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney2 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22