# import matplotlib.pyplot as plt
from MDlogger import plt
import numpy as np
import MDlogger

doc = MDlogger.Doc(filename = "test.md")
x = np.arange(0,10,0.5)
y = np.sin(x)
plt.plot(x,y)
# f = plt.gcf()
# f.savefig("aa.png")
# plt.plot(x,y)
doc.addplt()
doc.generate()