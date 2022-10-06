import numpy as np
from abc import abstractmethod

from models.math import Vector, Quaternion, Rotation


class Test:
    @abstractmethod
    def run(self):
        pass


class QuaternionRotationTest(Test):
    def run(self):
        face = Quaternion(0, 1, 0, 0)
        head = Quaternion(0, 0, 0, 1)

        rotation = Rotation(np.pi/2, 0, 1, 0)

        print(rotation)
        print(rotation.rotate(face))
        print(rotation.rotate(head))


class VectorOperationTest(Test):
    def run(self):
        """ Multiplication within an integer """
        v1 = Vector(np.random.random(), np.random.random(), np.random.random())
        m = np.random.randint(1, 5)
        print(v1, m, v1 * m, m * v1, v1*m == m*v1)

        """ Division within a float """
        print(v1, m, v1 / m)

        """ Addition """
        v2 = Vector(np.random.random(), np.random.random(), np.random.random())
        print(v1, v2, v1 + v2, v2 + v1)

        """ Subtraction """
        print(v1, v2, v1 - v2, v2 - v1)

        """ Projection """
        x = Vector(1, 1, 1)
        y = Vector(1, 0, 1)
        projection = x.projection(y)
        print(x)
        print(y)
        print(projection)
        print(x.scalar_projection(y))


if __name__ == '__main__':
    test = VectorOperationTest()
    test.run()
