import scipy.io as spio, numpy as np, pathlib, sys
src = pathlib.Path(r"C:\Users\mrjel\Downloads\doi_10_5061_dryad_2h7d2__v20160420 (1)\Data+FBG+group+delay.mat")
out = pathlib.Path("figures/recon_artifact_with_meas.npz")
found = {}
try:
    m = spio.loadmat(str(src))
    # common names to check; edit this tuple if inspect showed different names
    for k in ('i1','i2','I1','I2','i_1','i_2'):
        if k in m:
            arr = np.asarray(m[k]).ravel()
            found[k] = arr
            print("Found", k, "shape", arr.shape)
    # If nothing found with common names, list all non-hidden keys and save them (optional)
    if not found:
        keys = [k for k in m.keys() if not k.startswith('__')]
        print("No common i1/i2 names found. Available keys:", keys)
        # Save all numeric arrays found under those keys so you can inspect them locally
        for k in keys:
            try:
                a = np.asarray(m[k])
                # only save arrays with numeric dtype and size>1
                if a.size > 1 and np.issubdtype(a.dtype, np.number):
                    found[k] = a.ravel()
                    print("Auto-saved", k, "shape", a.ravel().shape)
            except Exception:
                pass
except Exception as e:
    print("Failed to load MAT file:", e)
if found:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **found)
    print("Saved measured traces to", out.resolve(), "keys:", list(found.keys()))
else:
    print("No numeric arrays saved from MAT file.")
