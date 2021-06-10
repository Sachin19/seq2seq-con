import sys
from collections import defaultdict
import numpy as np

diff = defaultdict(int)
with open(sys.argv[1]) as f1, open(sys.argv[2]) as f2, open(
    sys.argv[1] + ".samelen", "w"
) as f1w, open(sys.argv[2] + ".samelen", "w") as f2w:
    for l in f1:
        l2 = f2.readline()
        len1 = len(l.split())
        len2 = len(l2.split())
        diff[np.abs(len1 - len2)] += 1

        if abs(len1 - len2) < 1:
            f1w.write(l)
            f2w.write(l2)


print(diff)

import matplotlib.pyplot as plt

plt.style.use("ggplot")

x, y = zip(*sorted(list(diff.items()), key=lambda x: x[1]))

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color="green")
plt.xlabel("length diff")
plt.ylabel("count")

plt.xticks(x_pos, x)

plt.savefig("trash/len.png")