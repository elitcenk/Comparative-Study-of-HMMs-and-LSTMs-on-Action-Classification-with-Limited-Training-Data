REM python extractFeature.py -f CNN -exp deneyCNN1
python hmmTrain.py -s 3 -exp deneyCNN1 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22
python hmmTest.py -exp deneyCNN1 -tp person02 person03 person05 person06 person07 person08 person09 person10 person22