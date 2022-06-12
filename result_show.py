from matplotlib import pyplot as plt
f1 = open("rad_error.txt")
f2 = open("pixel_error.txt")
rad_error = f1.readlines()[0].split(", ")
rad_error = [float(x) for x in rad_error]
pixel_error = f2.readlines()[0].split(", ")
pixel_error = [float(x) for x in pixel_error]
plt.figure(figsize=(18,8),dpi=100)
plt.subplot(1,2,1)
plt.hist(rad_error,1000)
plt.title("radian error distribution\n the maximum radian error is %s"%(max(rad_error)))
plt.subplot(1,2,2)
plt.hist(pixel_error,1000)
plt.title("pixel error distribution\n the maximum pixel error is %s"%(max(pixel_error)))
plt.savefig("a.jpg")
plt.show()