python extractFeature.py -mhi -f Hu -exp deney13a
python hmmTrain.py -s 11 -exp deney13a -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney13a -tp person02 person03 person05 person06 person07 person08 person09 person10 person22