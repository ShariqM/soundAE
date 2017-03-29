import matplotlib.pyplot as plt
import time
import numpy as np


n_rows, n_cols = 2, 2
figure, axes = plt.subplots(n_rows, n_cols)

plots = []
for i in range(n_rows):
    for j in range(n_rows):
        plots.append(axes[i,j].plot(np.random.randn(10))[0])
plt.show(block=False)
time.sleep(2)

for t in range(10):
    for i in range(len(plots)):
        plots[i].set_data(range(10), np.random.randn(10))
        figure.canvas.draw()
        plt.show(block=False)
        for k in range(100000):
            k = k + 1
print ('done')
plt.show()
