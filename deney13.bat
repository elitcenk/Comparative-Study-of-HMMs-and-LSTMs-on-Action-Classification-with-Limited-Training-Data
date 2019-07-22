python extractFeature.py -mhi -f Hu -exp deney13
python hmmTrain.py -s 7 -exp deney13 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney13 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22