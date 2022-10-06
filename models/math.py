import numpy as np


class Vector:
    """ Vector: class of 3-dimensional vector for Coordinate and Direction """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def unpack(self):
        """ unpack: unpacks the elements of the vector """
        return self.x, self.y, self.z

    def update(self, x, y, z):
        """ update: updates the elements of the vector """
        self.x = x
        self.y = y
        self.z = z

    def vectorize(self):
        """ vectorize: returns list form of the vector, for concatenation with other lists """
        return [self.x, self.y, self.z]

    def dot(self, other):
        """ dot: performs dot product of vectors """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """ cross: performs cross product of vectors """
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def size(self):
        """ size: returns the size of the vector """
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))

    def projection(self, other):
        """ projection: performs projection on a plane defined by a tangent vector """
        assert isinstance(other, Vector)
        return (self.dot(other) / other.size()) * (other / other.size())

    def scalar_projection(self, other):
        """ scalar_projection: performs projection but returns the size of the projection vector only """
        assert isinstance(other, Vector)
        return self.dot(other) / other.size()

    def get_distance(self, other):
        """ get_distance: get distance between to vectors """
        assert isinstance(other, Vector)
        return (self - other).size()

    def get_cosine_angle(self, other):
        """ get_cosine_angle: calculates cosine value between the vector and target vector """
        assert isinstance(other, Vector)
        return self.dot(other) / (self.size() * other.size())

    def get_angle(self, other):
        """ get_angle: calculates angle between the vector and target vector, in degree scale """
        return np.degrees(np.arccos(self.get_cosine_angle(other)))

    def to_unit(self):
        """ to_unit: converts the vector to a unit vector that is parallel to the vector but size 1 """
        denominator = np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))
        return Vector(x=self.x/denominator, y=self.y/denominator, z=self.z/denominator)

    def to_quaternion(self):
        """ to_quaternion: convert the vector to quaternion which w value is 0 """
        return Quaternion(0, self.x, self.y, self.z)

    def __str__(self):
        return "(X:{x}, Y:{y}, Z:{z})".format(x=self.x, y=self.y, z=self.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other):
        return Vector(other.x + self.x, other.y + self.y, other.z + self.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other):
        return Vector(other.x - self.x, other.y - self.y, other.z - self.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return Vector(other * self.x, other * self.y, other * self.z)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other):
        return isinstance(other, Vector) and self.x == other.x and self.y == other.y and self.z == other.z


class Quaternion:
    """ Quaternion: class of 4-dimensional quaternion for Orientation and Rotation """
    def __init__(self, w, i, j, k):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def unpack(self):
        """ unpack: unpacks the elements of the quaternion """
        return self.w, self.i, self.j, self.k

    def update(self, w, i, j, k):
        """ update: updates the elements of the quaternion """
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def vectorize(self):
        """ vectorize: returns list form of the quaternion, , for concatenation with other lists """
        return [self.w, self.i, self.j, self.k]

    def get_vector_part(self):
        """ get_vector_part: returns only vector part of the quaternion as an instance of Vector class """
        return Vector(self.i, self.j, self.k)

    def get_scalar_part(self):
        """ get_scalar_part: returns only scalar part of the quaternion """
        return self.w

    def get_conjugate(self):
        """ get_conjugate: returns conjugate quaternion of the quaternion """
        return Quaternion(self.w, -self.i, -self.j, -self.k)

    def is_unit(self):
        """ is_unit: examines whether the size of the quaternion is 1 or not """
        return np.square(self.w) + np.square(self.i) + np.square(self.j) + np.square(self.k) == 1.

    def __str__(self):
        return "(W:{w}, I:{i}, J:{j}, K:{k})".format(w=self.w, i=self.i, j=self.j, k=self.k)

    def __mul__(self, other):
        assert isinstance(other, Quaternion)
        return Quaternion(
            w=self.w*other.w - self.i*other.i - self.j*other.j - self.k*other.k,
            i=self.w*other.i + self.i*other.w + self.j*other.k - self.k*other.j,
            j=self.w*other.j - self.i*other.k + self.j*other.w + self.k*other.i,
            k=self.w*other.k + self.i*other.j - self.j*other.i + self.k*other.w
        )

    def __rmul__(self, other):
        assert isinstance(other, Quaternion)
        return Quaternion(
            w=other.w*self.w - other.i*self.i - other.j*self.j - other.k*self.k,
            i=other.w*self.i + other.i*self.w + other.j*self.k - other.k*self.j,
            j=other.w*self.j - other.i*self.k + other.j*self.w + other.k*self.i,
            k=other.w*self.k + other.i*self.j - other.j*self.i + other.k*self.w
        )


class Rotation(Quaternion):
    """ Rotation: class of rotation quaternion, receives axis and angle, then construct unit rotation vector """
    def __init__(self, theta, i, j, k):
        assert -2 * np.pi <= theta <= 2 * np.pi  # theta is radian

        """ Rotation should be a unit vector """
        denominator = np.square(i) + np.square(j) + np.square(k)
        Quaternion.__init__(
            self,
            w=np.cos(theta / 2),
            i=np.sign(i) * np.sqrt(np.square(i) / denominator) * np.sin(theta / 2),
            j=np.sign(j) * np.sqrt(np.square(j) / denominator) * np.sin(theta / 2),
            k=np.sign(k) * np.sqrt(np.square(k) / denominator) * np.sin(theta / 2)
        )

    def rotate(self, quaternion):
        return self * quaternion * self.get_conjugate()