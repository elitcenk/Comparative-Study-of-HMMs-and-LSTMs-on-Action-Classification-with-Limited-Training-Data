python extractFeature.py -f Hu -exp deney12
python hmmTrain.py -s 7 -exp deney12 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney12 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22