import numpy as np, os, sys
p = os.path.join("data","raw","pipeline_input.npz")
if not os.path.exists(p):
    print("pipeline_input.npz not found at", p); sys.exit(1)
d = np.load(p, allow_pickle=True)
print("pipeline_input.npz keys:", d.files)
for k in d.files:
    print(k, d[k].shape, d[k].dtype)
