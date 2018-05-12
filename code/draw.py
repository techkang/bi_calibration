import matplotlib.pyplot as plt

plt.scatter((-3,3,-2.25,2.25,0),(-1,-1,0,0,3))

plt.plot((-3,0),(-1,3),color='b')
plt.plot((3,0),(-1,3),color='b')
plt.plot((3,-3),(-1,-1),color='b')
plt.plot((-2.25,2.25),(0,0),color='b')

plt.plot((-2.25,-2.25),(0,-1),'--',color='black')
plt.plot((0,0),(3,-1),'--',color='black')

plt.text(1,-0.9,'b',fontdict={'size':16})
plt.text(-1, 0.2,'b-|d|',fontdict={'size':16})

plt.annotate(r'f', xy=(-2.25, -0.4), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.annotate(r'z' , xy=(0, 1), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.annotate(r'O' , xy=(-3, -1), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.annotate(r"O'" , xy=(3, -1), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.annotate(r'$p_l$' , xy=(-2.25, 0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.annotate(r'$p_r$' , xy=(2.25, 0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.annotate(r'$X$' , xy=(0, 3), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.axis('off')

plt.show()