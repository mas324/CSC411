def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]
    if z < 0:
        return -1
    else:
        return 1