import matplotlib.pyplot as plt

data = {}
data['cuisine']= [0.976684, 0.970207, 0.963758, 0.950142, 0.646417, 0.404553, 0.333919, 0.297012]
data['location']= [0.980794, 0.959027, 0.815508, 0.445802, 0.294014, 0.262324, 0.246479, 0.258803]
data['size']= [1, 1, 0.995957, 0.819383, 0.577488, 0.521739, 0.498239, 0.484155]
data['price']= [1, 1, 1, 0.997122, 0.984035, 0.919762, 0.781847, 0.649913]

axis = [0,1,2,3,4,5,6,7]

plt.plot(axis, data["cuisine"], "r", label="cuisine")
plt.plot(axis, data["location"], "b", label="location")
plt.plot(axis, data["size"], "g",label="party size")
plt.plot(axis, data["price"], "y", label="price range")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('relative position')
plt.show()