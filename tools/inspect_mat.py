import scipy.io as spio, pathlib, sys
p = pathlib.Path(r"C:\Users\mrjel\Downloads\doi_10_5061_dryad_2h7d2__v20160420 (1)\Data+FBG+group+delay.mat")
try:
    info = spio.whosmat(str(p))
    names = [name for name, shape, dtype in info]
    print("Variables in", p.name, ":", names)
except Exception as e:
    print("Failed to inspect MAT file:", e)
    sys.exit(2)
