import time
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
#from skimage.restoration import unwrap_phase

import scalardiffract as sdt


##########################################################################
# Parameters
##########################################################################
    
# Wavelength in metres
#wl = 1550e-9

# Number of DOE pixels
n_src = 512

# Pixel pitch of the DOE in wavelength units
p_src = 1200.0 / n_src

# Image oversampling factor
over = 6.0

# Image position in wavelength units
z_img = 50000.0


##########################################################################
# Main program
##########################################################################

# Source pixel grid
x_src, y_src = sdt.grid(p_src, n_src)

# Source size
s_src = n_src * p_src
print("Source size (quadratic): %.1f λ (%d px)" % (s_src, n_src))

# Spherical wave
f = z_img
ratio = 0.5
u_src = sdt.spherical_wave(f, x_src, y_src)
print("Lens focus: %.1f λ" % f)

# Wavefront test of source field
fc, rc = sdt.get_focus(u_src, x_src, y_src)
fc_flat = np.array(fc.flat)
i = np.argwhere(np.logical_not(np.isnan(fc_flat)))
df = fc_flat[i]/f - 1.0
print("Focus deviation (min): %.1e" % np.min(df))
print("Focus deviation (max): %.1e" % np.max(df))

# Apply circular aperture
r = 0.49 * (n_src * p_src)
print("Aperture diameter: %.1f λ" % (2*r))
aperture_mask = sdt.circle_mask(r, x_src, y_src)
#aperture_mask = np.ones((n_src, n_src), dtype=float)
u_src *= aperture_mask

# Normalize source field
u_src /= np.sqrt(np.sum(np.abs(u_src)**2)) * p_src

# Distance of image plane from DOE in wavelength units
print("Image distance: %.1f λ (%.2f f)" % (z_img, z_img/f))

# Optimum image parameters
s_img = 1200.0
p_img, n_img, fraction = sdt.opt_params(z_img, p_src, n_src, over, s_img=s_img)
print("Image oversampling: %.2f" % over)
print("Image fraction: %.2f" % fraction)

# Image pixel grid
x_img, y_img = sdt.grid(p_img, n_img)

# Image size
s_img = n_img * p_img
print("Image size (quadratic): %.1f λ (%d px)" % (s_img, n_img))

# Rayleigh-Sommerfeld method
t = time.time()
u_img = sdt.rs(z_img, p_src, p_img, u_src, n_img)
t = time.time() - t
print("Calculation time: %.1f s" % t)

# Power of source and image fields
pwr_src = sdt.wave_power(u_src, p_src)
pwr_img = sdt.wave_power(u_img, p_img)
print("Power efficiency: %.1f %%" % (100 * (pwr_img / pwr_src)))

# Beam size
wx, wy = sdt.beam_width(u_img, x_img, y_img, ratio)
if wx is None:
    print("Image beam diameter: ---")
else:
    print("Image beam diameter: %.1f λ" % wx)


##########################################################################
# Show source and image fields
##########################################################################

fig, axes = plt.subplots(2,3, figsize=(14, 8))

ax = axes[0,0]
ax.set_title("Source Magnitude [rel.]")
data = np.abs(u_src)
data /= np.max(data)
img = ax.imshow(data, interpolation='nearest')
plt.colorbar(img, fraction=0.046, pad=0.04)

ax = axes[0,1]
ax.set_title("Source Phase [π]")
data = np.angle(u_src)
data /= np.pi
img = ax.imshow(data, vmin=-1.0, vmax=1.0, interpolation='nearest')
ticks = np.arange(-1.0, 1.1, 0.5)
plt.colorbar(img, fraction=0.046, pad=0.04, ticks=ticks)

ax = axes[0,2]
#ax.axis("off")
ax.set_title("Source Magnitude [rel.]")
data = np.abs(u_src)
data = data[data.shape[0]//2,:]
data /= np.max(data)
ax.plot(data)


ax = axes[1,0]
ax.set_title("Image Magnitude [rel.]")
data = np.abs(u_img)
data /= np.max(data)
img = ax.imshow(data, interpolation='nearest')
plt.colorbar(img, fraction=0.046, pad=0.04)

ax = axes[1,1]
ax.set_title("Image Phase [π]")
data = np.angle(u_img)
data /= np.pi
img = ax.imshow(data, vmin=-1.0, vmax=1.0, interpolation='nearest')
ticks = np.arange(-1.0, 1.1, 0.5)
plt.colorbar(img, fraction=0.046, pad=0.04, ticks=ticks)

ax = axes[1,2]
ax.set_title("Image Magnitude [rel.]")
data = np.abs(u_img)
data = data[data.shape[0]//2,:]
data /= np.max(data)
ax.plot(data)

plt.savefig("sphere-%d.png" % n_src)
plt.show()
