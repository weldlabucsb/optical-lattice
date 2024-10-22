import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import optical as op
import band
import pickle


##test of class with four waves:
#we set lambda = 1 = c thus lambda = 1, kmag = 2*pi, w = 1/(2*pi)
#k vectors:
kmag = 2*np.pi
# k1 = kmag*np.array([1,0,0])
# k2 = kmag*np.array([-1,0,0])
# k3 = kmag*np.array([0,1,0])
# k4 = kmag*np.array([0,-1,0])

#or you can use a function to get the kvectors from an angle:
k1 = kmag*op.Epwave.angle2kvector(0)
k2 = kmag*op.Epwave.angle2kvector(90)
k3 = kmag*op.Epwave.angle2kvector(180)
k4 = kmag*op.Epwave.angle2kvector(270)
#amplitudes:
#e1, e2, e3, e4 = 0.6,1,0.55,1
e1, e2, e3, e4 = 1,1,0.6, 0.5
#phases:
# ph = [0, 90, 0, -75, 45]
#ph = [0, 0, 90, -70, 45]
ph = [0, 0, 90, -70, 0]
ph1, ph2, ph3, ph4, ph5 = list(map(lambda x: x*np.pi/180, ph))
#polarization:
p1,p2,p3,p4 = np.array([0,0,1]),np.array([0,0,1]), np.array([0,0,1]), np.array([0,0,1])
#angular freq:
w = 1/(2*np.pi)

#initialization of Ep waves:
E1 = op.Epwave(e1, p1, w, k1, ph1)
E2 = op.Epwave(e2, p2, w, k2, ph2)
E3 = op.Epwave(e3, p3, w, k3, ph3)
E4 = op.Epwave(e4, p4, w, k4, ph4)
E5 = op.Epwave(e4, p4, w, k4, ph5)
#print angles:
print(f"E1: {E1.angle()}, E2: {E2.angle()}, E3: {E3.angle()}, E4: {E4.angle():.1f}")
#creation of the wave packet.
Sum = E1+E2+E3+E4
#rotates all the beams in the wave packet.
#Sum.rotate_beams(-45, [0,0,1]) #rotate the beams by 45 degrees around the z axis.
#print angles:
print(f"E1: {E1.angle()}, E2: {E2.angle()}, E3: {E3.angle()}, E4: {E4.angle():.1f}")
X = np.arange(3, step=0.01)
Y = np.arange(3,  step=0.01)

xx, yy = np.meshgrid(X, Y)
zz = np.zeros(xx.shape)

Z = Sum.tavg_intensity(xx, yy, 0, savefile="intensity_2D") #potential
print(f"depth = {np.max(Z):.1e}")
Y1 = np.linspace(0,1,2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for y in Y1:
#     pot2 = lambda x: Sum.tavg_intensity(x, y, 0)
#     POT = pot2(X)
    
#     ax.plot(X, POT)


print(f"value = {Z}")

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
im = ax1.pcolormesh(xx, yy, Z)
ax1.set_title("Potential for the Kitaev Chain")
ax1.set_xlabel("Position x")
ax1.set_ylabel("Position y")
fig1.colorbar(im, ax=ax1)
plt.show(block=False)

X = np.linspace(0,2, 100)
Ey = []
Q = []
X5 = np.linspace(0,2, 150)
fig3, ax3 = plt.subplots()
counter = 0
direction = np.array([1,1])
# for y in X:
#     print(f"Calculating bands for y = {y}. {len(X)-counter} bands missing")
    
#     pot_y = lambda vec: Sum.tavg_intensity(vec[0], vec[1], 0)
#     _band = band.BandSolver(pot_y, 5, kx=kmag, ky=kmag, dim=2)

#     #ax3.plot(X5, pot_y(X5), label = f"y = {y}")

#     Q, E, V = _band.solve(-1*kmag, 1*kmag, direction=direction, N=30)
#     Ey.append(E)
#     counter += 1
pot_y = lambda vec: Sum.tavg_intensity(vec[0], vec[1], 0)
_band = band.BandSolver(pot_y, V0=25, jmax=3, kx=kmag, ky=kmag, dim=2)
qx = np.linspace(-1*kmag, kmag, 20)
xx, yy = np.meshgrid(qx, qx)
def calculate_band_structure(qxx, qyy):
    zz = np.empty((qxx.shape[0], qxx.shape[1], (2*_band.jmax+1)**2))
    pairs = qxx.shape[0]*qyy.shape[0]
    print(f"Started calculating the energy spectra for {pairs} pairs of qx, qy.")
    counter = 0
    for i in range(qxx.shape[0]):
        for j in range(qyy.shape[0]):
            eigen, v = _band._solve2D(np.array([qxx[i,j], qyy[i,j]]))
            perm = np.argsort(eigen)
            zz[i, j, :] = eigen
            counter += 1
            if not counter%100:
                print(f"Calculated {counter} points out of {pairs}. ({counter/pairs*100:.1f} %)")
    return zz
zz = calculate_band_structure(xx, yy)



#ax3.plot(X5, pot_y(X5), label = f"y = {y}")

#Q, E, V = _band.solve(-1*kmag, 1*kmag, direction=direction, N=30)

# Ey = np.moveaxis(np.array(Ey), 1, 2) #moves the axis such as first axis is X pos, second is Q value, and last is the bands n number

ax3.legend()
X1, Q1 = np.meshgrid(X, Q)
Z = np.zeros((len(X), len(Q)))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
# for i in range(len(X)):
#     for band in Ex[i,:,:]:
#         Z[i] = Ex[i,:,0]
ax2.set_xlabel("Qx")
ax2.set_ylabel("Quasi momemtum qy, k = 2*pi")
ax2.set_zlabel("Energy/Er")
ax2.set_title("Band Structure (first 4 bands) for duck potential")
maxval = np.max(zz[:,:,:4])
minval = np.min(zz[:,:,:4])
for i in range(4):
    im = ax2.plot_surface(xx,yy, zz[:,:, i], label = f"n = {i}", cmap="magma", vmin = minval, vmax=maxval)
fig2.colorbar(im, ax=ax2)
# with open("bands_y", 'ab') as file:
#     print("file saved")
#     pickle.dump(Ey, file)




# X = np.linspace(0,2, 100)


# def sin_potential(x, V0 = 1.34, ph = 2.345,k = kmag, V1=0.72):
#     return V0*np.power(np.sin(k*x+ph),2)+V1

# sin_band = band.BandSolver(sin_potential, jmax=20, k=kmag)
# Qs, Es, Vs = sin_band.solve(-1*kmag, 1*kmag)
# SIN_P = sin_potential(X)
# # ax3.plot(X, SIN_P, label = "sinusoidal potential")
# ax3.legend()

# for i in range(2):
#     ax2.plot(Qs, Es[i], label=f"sin, n = {i}")
# ax2.legend()
# counter = 0
# nmax = 5
# for band in E:
#     if counter >nmax:
#         break
#     ax2.plot(Q, band)
#     counter += 1

plt.show()


#plt.show()