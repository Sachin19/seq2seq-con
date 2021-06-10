import sys
from collections import defaultdict
import numpy as np

diff = defaultdict(int)
with open(sys.argv[1]) as f:
    for l in f:
        aligns = l.strip().split()
        for align in aligns:
            pos1, pos2 = align.split("-")
            diff[np.abs(int(pos1) - int(pos2))] += 1

print(diff)

import matplotlib.pyplot as plt

plt.style.use("ggplot")

x, y = zip(*sorted(list(diff.items()), key=lambda x: x[1]))

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color="green")
plt.xlabel("Alignment distance")
plt.ylabel("Count")

plt.xticks(x_pos, x)

plt.savefig("trash/align.png")