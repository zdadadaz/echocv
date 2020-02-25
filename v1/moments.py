import numpy as np

def magvect(v):
  magnitude = np.sqrt(v[0]*v[0] + v[1]*v[1])
  return magnitude

# angle between vectors
def angle(v1,v2):
  ang = np.arccos(np.dot(v1,v2)/(magvect(v1)*magvect(v2)))
  print(ang)
  ang = ang*180/np.pi
  #if (ang > 90):
  #    ang = ang - 180
  #else:
  #    ang = ang
  return ang

def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def inertial_axis(data):
    """Calculate the x-mean, y-mean, and cov matrix of an image."""
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov

def plot_bars(x_bar, y_bar, cov, ax):
    """Plot bars with a length of 2 stddev along the principal axes."""
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='white')
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color='red')
    ax.axis('image')
    
def extractparameters(x_bar, y_bar, cov):
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    out1 = make_lines(eigvals, eigvecs, mean, -1)
    height = (out1[0][0] - out1[0][2], out1[1][0] - out1[1][2])
    center = (out1[0][1], out1[1][1])
    out2 = make_lines(eigvals, eigvecs, mean, 0)
    width = (out2[0][0] - out2[0][2], out2[1][0] - out2[1][2])
    theta = angle(height, (0, 1))
    return magvect(height), magvect(width), theta, center, out1, out2

