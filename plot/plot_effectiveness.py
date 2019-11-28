""" Plot visual service effectiveness model on 2-dimensional space """

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from models.entity import User, Device, Service
from models.orientation import generate_random_orientation
from models.mobility import Direction, Coordinate, StaticMobility, generate_horizontal_direction_specific_speed_mobility, generate_custom_mobility
from models.orientation import Orientation
from models.effectiveness import VisualEffectiveness


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


fig = plt.figure()
ax = fig.gca(projection='3d')

width = 10
height = 10
depth = 2
delta = 1

effectiveness = VisualEffectiveness(text_size_pixel=12,
                                    resolution=1080,
                                    visual_angle_min=5/60,
                                    FoV_angle_max=105,
                                    face_angle_max=60)

device = Device(name=0,
                device_type="visual",
                coordinate=Coordinate(x=0, y=0, z=1),
                mobility=StaticMobility(),
                orientation=Orientation(theta=-np.pi/2, i=0, j=0, k=1),
                size=1)
service = Service(name=0,
                  service_type="visual",
                  device=device)

X = np.arange(-width, width, delta)
Y = np.arange(-height, height, delta)
Z = []  # effectiveness

for y in Y:
    temp = []
    for x in X:
        if x == 0 and y == 0:
            temp.append(1)
        else:
            user = User(uid=0,
                        # Start from edge of the environment
                        coordinate=Coordinate(x=x, y=y, z=1.7),
                        mobility=generate_custom_mobility(width, height, depth, Direction(-x, -y, 0), 1))
                        # mobility=generate_horizontal_direction_specific_speed_mobility(width, height, depth, 1))
            temp.append(effectiveness.measure(user, service))
    Z.append(temp)
Z = np.array(Z)
X, Y = np.meshgrid(X, Y)

service_direction = Arrow3D([0, service.device.orientation.face.i*5],
                            [0, service.device.orientation.face.j*5],
                            [0, service.device.orientation.face.k],
                            mutation_scale=20, lw=3, arrowstyle="-|>", color="r", zorder=10)
ax.add_artist(service_direction)

ax.set_zlim(-0.01, 1.0)

# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
scatter = ax.scatter(X, Y, Z)
# wire = ax.plot_wireframe(X, Y, Z)
# contour = ax.contourf(X, Y, Z)

# x, y, z = -3, 3, 1.7
# example_user = User(uid=0,
#                     coordinate=Coordinate(x=x, y=y, z=z),
#                     mobility=generate_custom_mobility(width, height, depth, Direction(-x, -y, 0), 0))
# print(service.device.orientation.face.get_vector_part())
# user_direction = Arrow3D([x, x+example_user.orientation.get_vector_part().x*5],
#                          [y, y+example_user.orientation.get_vector_part().y*5],
#                          [1.5, 1.5 + example_user.orientation.get_vector_part().z*5],
#                          mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
# ax.add_artist(user_direction)
# print(example_user.orientation.get_vector_part())
# print(effectiveness.measure(example_user, service))

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

plt.draw()
plt.show()

