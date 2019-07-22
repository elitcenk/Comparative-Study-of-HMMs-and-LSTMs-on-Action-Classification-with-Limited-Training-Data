python extractFeature.py -f Hu -exp deney3
python hmmTrain.py -s 3 -exp deney3 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deney3 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22