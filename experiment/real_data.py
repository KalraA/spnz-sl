from kfold import run_ilspn

T = True
F = False
N = "normal"
arr = [2, 3, 4]

run_ilspn("real", F, "data/real/abalone", 'abalone',  8,   2, 100, N)
#run("real", F, 'banknote',    4,   2,   1,  10, 0.1, T, T, 0, N)
#run("real", F, 'sensorless', 48,   2, 256, 256, 0.1, T, T, 4, N)
#run("real", F, 'flowdata',    3,   2, 256, 256, 0.1, T, T, 0, N)
#run("real", F, 'abalone',     8,   2,   1,  10, 0.1, T, T, 4, N)
#run("real", F, 'ki',          8,   2,  10,  10, 0.1, T, T, 4, N)
#run("real", F, 'ca',         22,   2,  10,  10, 0.1, T, T, 4, N)


