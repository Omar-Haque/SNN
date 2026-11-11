"""extract.py SRC OUT [EXAMPLES] -- write a code_eval-loadable layer 1 (W, theta) from a train_l1 output, optionally at a frame."""
import sys, numpy as np
src, out = sys.argv[1], sys.argv[2]; d = np.load(src)
if len(sys.argv) > 3:
    e = int(sys.argv[3]); fi = int(np.argmin(np.abs(d["frame_examples"] - e))); si = int(np.argmin(np.abs(d["snap_examples"] - e)))
    W, th = d["frames"][fi], d["snap_theta"][si]; print(f"frame at {d['frame_examples'][fi]}, theta at {d['snap_examples'][si]}")
else:
    W, th = d["W"], d["theta"]
np.savez(out, W=W, theta=th, patch=d["patch"], stride=d["stride"])
