# -*- coding: utf-8 -*-
# 
# cython: boundscheck=False
# cython: wraparound=False
#
import quaternion # noqa
import math
import numpy as np
from math import sin, cos, acos, atan2, asin, pi, sqrt, degrees, radians

from mmd.utils.MLogger import MLogger # noqa

logger = MLogger(__name__)


class MRect:

    def __init__(self, x=0, y=0, width=0, height=0):
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def width(self):
        return self.__width

    def height(self):
        return self.__height

    def __str__(self):
        return "MRect({0}, {1}, {2}, {3})".format(self.__x, self.__y, self.__width, self.__height)


class MVector2D:

    def __init__(self, x=0.0, y=0.0):
        if isinstance(x, float):
            # 実数の場合
            self.__data = np.array([x, y], dtype=np.float64)
        elif isinstance(x, MVector2D):
            # クラスの場合
            self.__data = np.array([x.x(), x.y()], dtype=np.float64)
        elif isinstance(x, np.ndarray):
            # arrayそのものの場合
            self.__data = np.array([x[0], x[1]], dtype=np.float64)
        else:
            self.__data = np.array([x, y], dtype=np.float64)

    def length(self):
        return float(np.linalg.norm(self.data(), ord=2))

    def lengthSquared(self):
        return float(np.linalg.norm(self.data(), ord=2)**2)

    def normalized(self):
        l2 = np.linalg.norm(self.data(), ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        normv = self.data() / l2
        return MVector2D(normv[0], normv[1])

    def normalize(self):
        l2 = np.linalg.norm(self.data(), ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        self.__data /= l2
    
    def effective(self):
        self.__data[np.isnan(self.data())] = 0
        self.__data[np.isinf(self.data())] = 0

        return self
            
    def data(self):
        return self.__data

    def __str__(self):
        return "MVector2D({0}, {1})".format(self.x(), self.y())

    def __lt__(self, other):
        return np.all(np.less(self.data(), other.data()))

    def __le__(self, other):
        return np.all(np.less_equal(self.data(), other.data()))

    def __eq__(self, other):
        return np.all(np.equal(self.data(), other.data()))

    def __ne__(self, other):
        return np.any(np.not_equal(self.data(), other.data()))

    def __gt__(self, other):
        return np.all(np.greater(self.data(), other.data()))

    def __ge__(self, other):
        return np.all(np.greater_equal(self.data(), other.data()))

    def __add__(self, other):
        if isinstance(other, np.float):
            v = self.add_float(other)
        elif isinstance(other, MVector2D):
            v = self.add_MVector2D(other)
        elif isinstance(other, np.int):
            v = self.add_int(other)
        else:
            v = self.data() + other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def add_MVector2D(self, other):
        return self.__data + other.__data

    def add_float(self, other: float):
        return self.__data + other

    def add_int(self, other: int):
        return self.__data + other

    def __sub__(self, other):
        if isinstance(other, np.float):
            v = self.sub_float(other)
        elif isinstance(other, MVector2D):
            v = self.sub_MVector2D(other)
        elif isinstance(other, np.int):
            v = self.sub_int(other)
        else:
            v = self.data() - other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def sub_MVector2D(self, other):
        return self.__data - other.__data

    def sub_float(self, other: float):
        return self.__data - other

    def sub_int(self, other: int):
        return self.__data - other

    def __mul__(self, other):
        if isinstance(other, np.float):
            v = self.mul_float(other)
        elif isinstance(other, MVector2D):
            v = self.mul_MVector2D(other)
        elif isinstance(other, np.int):
            v = self.mul_int(other)
        else:
            v = self.data() * other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def mul_MVector2D(self, other):
        return self.__data * other.__data

    def mul_float(self, other: float):
        return self.__data * other

    def mul_int(self, other: int):
        return self.__data * other

    def __truediv__(self, other):
        if isinstance(other, np.float):
            v = self.truediv_float(other)
        elif isinstance(other, MVector2D):
            v = self.truediv_MVector2D(other)
        elif isinstance(other, np.int):
            v = self.truediv_int(other)
        else:
            v = self.data() / other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def truediv_MVector2D(self, other):
        return self.__data / other.__data

    def truediv_float(self, other: float):
        return self.__data / other

    def truediv_int(self, other: int):
        return self.__data / other

    def __floordiv__(self, other):
        if isinstance(other, np.float):
            v = self.floordiv_float(other)
        elif isinstance(other, MVector2D):
            v = self.floordiv_MVector2D(other)
        elif isinstance(other, np.int):
            v = self.floordiv_int(other)
        else:
            v = self.data() // other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def floordiv_MVector2D(self, other):
        return self.__data // other.__data

    def floordiv_float(self, other: float):
        return self.__data // other

    def floordiv_int(self, other: int):
        return self.__data // other

    def __mod__(self, other):
        if isinstance(other, np.float):
            v = self.mod_float(other)
        elif isinstance(other, MVector2D):
            v = self.mod_MVector2D(other)
        elif isinstance(other, np.int):
            v = self.mod_int(other)
        else:
            v = self.data() % other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def mod_MVector2D(self, other):
        return self.__data % other.__data

    def mod_float(self, other: float):
        return self.__data % other

    def mod_int(self, other: int):
        return self.__data % other

    def __lshift__(self, other):
        if isinstance(other, MVector2D):
            v = self.data() << other.data()
        else:
            v = self.data() << other
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __rshift__(self, other):
        if isinstance(other, MVector2D):
            v = self.data() >> other.data()
        else:
            v = self.data() >> other
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __and__(self, other):
        v = self.data() & other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __dataor__(self, other):
        v = self.data() ^ other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __or__(self, other):
        v = self.data() | other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __neg__(self):
        return self.__class__(-self.x(), -self.y())

    def __pos__(self):
        return self.__class__(+self.x(), +self.y())

    def x(self):
        return self.__data[0]

    def y(self):
        return self.__data[1]
    
    def setX(self, x):
        self.__data[0] = x

    def setY(self, y):
        self.__data[1] = y

    def to_log(self):
        return "x: {0}, y: {1}".format(round(self.x(), 5), round(self.y(), 5))


class MVector3D:

    def __init__(self, x=0.0, y=0.0, z=0.0):
        if isinstance(x, float):
            # 実数の場合
            self.__data = np.array([x, y, z], dtype=np.float64)
        elif isinstance(x, MVector3D):
            # クラスの場合
            self.__data = np.array([x.x(), x.y(), x.z()], dtype=np.float64)
        elif isinstance(x, np.ndarray):
            # arrayそのものの場合
            self.__data = np.array([x[0], x[1], x[2]], dtype=np.float64)
        else:
            self.__data = np.array([x, y, z], dtype=np.float64)

    def copy(self):
        return MVector3D(self.x(), self.y(), self.z())

    def length(self):
        return float(np.linalg.norm(self.data(), ord=2))

    def lengthSquared(self):
        return float(np.linalg.norm(self.data(), ord=2)**2)

    def normalized(self):
        l2 = np.linalg.norm(self.data(), ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        normv = self.data() / l2
        return MVector3D(normv[0], normv[1], normv[2])

    def normalize(self):
        self.effective()
        l2 = np.linalg.norm(self.data(), ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        self.__data /= l2
    
    def distanceToPoint(self, v):
        return MVector3D(self.data() - v.data()).length()
    
    def project(self, modelView, projection, viewport: MRect):
        tmp = MVector4D(self.x(), self.y(), self.z(), 1)
        tmp = projection * modelView * tmp
        if is_almost_null(tmp.w()):
            tmp.setW(1)

        tmp /= tmp.w()
        tmp = tmp * 0.5 + MVector4D(0.5, 0.5, 0.5, 0.5)
        tmp.setX(tmp.x() * viewport.width() + viewport.x())
        tmp.setY(tmp.y() * viewport.height() + viewport.y())

        tmp.effective()
        return tmp.toVector3D()

    def unproject(self, modelView, projection, viewport: MRect):
        inverse = (projection * modelView).inverted()

        tmp = MVector4D(self.x(), self.y(), self.z(), 1)
        tmp.setX((tmp.x() - viewport.x()) / viewport.width())
        tmp.setY((tmp.y() - viewport.y()) / viewport.height())
        tmp = tmp * 2 - MVector4D(1, 1, 1, 1)
        tmp.effective()

        obj = inverse * tmp
        if is_almost_null(obj.w()):
            obj.setW(1)

        obj /= obj.w()
        obj.effective()
        
        return obj.toVector3D()
        
    def toVector4D(self):
        return MVector4D(self.__data[0], self.__data[1], self.__data[2], 0)

    def is_almost_null(self):
        return (is_almost_null(self.__data[0]) and is_almost_null(self.__data[1]) and is_almost_null(self.__data[2]))
    
    def effective(self):
        self.__data[np.isnan(self.data())] = 0
        self.__data[np.isinf(self.data())] = 0

        return self
                
    def abs(self):
        self.setX(abs(get_effective_value(self.x())))
        self.setY(abs(get_effective_value(self.y())))
        self.setZ(abs(get_effective_value(self.z())))

        return self
                
    def one(self):
        self.effective()
        self.setX(1 if is_almost_null(self.x()) else self.x())
        self.setY(1 if is_almost_null(self.y()) else self.y())
        self.setZ(1 if is_almost_null(self.z()) else self.z())

        return self
    
    def non_zero(self):
        self.effective()
        self.setX(0.0000001 if is_almost_null(self.x()) else self.x())
        self.setY(0.0000001 if is_almost_null(self.y()) else self.y())
        self.setZ(0.0000001 if is_almost_null(self.z()) else self.z())

        return self
    
    def isnan(self):
        self.__data = self.data().astype(np.float64)
        return np.isnan(self.data()).any()

    @classmethod
    def crossProduct(cls, v1, v2):
        return crossProduct_MVector3D(v1, v2)

    @classmethod
    def dotProduct(cls, v1, v2):
        return dotProduct_MVector3D(v1, v2)
        
    def data(self):
        return self.__data

    def to_log(self):
        return "x: {0}, y: {1} z: {2}".format(round(self.__data[0], 5), round(self.__data[1], 5), round(self.__data[2], 5))

    def __str__(self):
        return "MVector3D({0}, {1}, {2})".format(self.__data[0], self.__data[1], self.__data[2])

    def __lt__(self, other):
        return np.all(np.less(self.data(), other.data()))

    def __le__(self, other):
        return np.all(np.less_equal(self.data(), other.data()))

    def __eq__(self, other):
        d1 = self.data()
        d2 = other.data()
        return d1[0] == d2[0] and d1[1] == d2[1] and d1[2] == d2[2]

    def __ne__(self, other):
        d1 = self.data()
        d2 = other.data()
        return d1[0] != d2[0] or d1[1] != d2[1] or d1[2] != d2[2]

    def __gt__(self, other):
        return np.all(np.greater(self.data(), other.data()))

    def __ge__(self, other):
        return np.all(np.greater_equal(self.data(), other.data()))

    def __add__(self, other):
        if isinstance(other, np.float):
            v = self.add_float(other)
        elif isinstance(other, MVector3D):
            v = self.add_MVector3D(other)
        elif isinstance(other, np.int):
            v = self.add_int(other)
        else:
            v = self.data() + other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def add_MVector3D(self, other):
        return self.__data + other.__data

    def add_float(self, other: float):
        return self.__data + other

    def add_int(self, other: int):
        return self.__data + other

    def __sub__(self, other):
        if isinstance(other, np.float):
            v = self.sub_float(other)
        elif isinstance(other, MVector3D):
            v = self.sub_MVector3D(other)
        elif isinstance(other, np.int):
            v = self.sub_int(other)
        else:
            v = self.data() - other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def sub_MVector3D(self, other):
        return self.__data - other.__data

    def sub_float(self, other: float):
        return self.__data - other

    def sub_int(self, other: int):
        return self.__data - other

    def __mul__(self, other):
        if isinstance(other, np.float):
            v = self.mul_float(other)
        elif isinstance(other, MVector3D):
            v = self.mul_MVector3D(other)
        elif isinstance(other, np.int):
            v = self.mul_int(other)
        else:
            v = self.data() * other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def mul_MVector3D(self, other):
        return self.__data * other.__data

    def mul_float(self, other: float):
        return self.__data * other

    def mul_int(self, other: int):
        return self.__data * other

    def __truediv__(self, other):
        if isinstance(other, np.float):
            v = self.truediv_float(other)
        elif isinstance(other, MVector3D):
            v = self.truediv_MVector3D(other)
        elif isinstance(other, np.int):
            v = self.truediv_int(other)
        else:
            v = self.data() / other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def truediv_MVector3D(self, other):
        return self.__data / other.__data

    def truediv_float(self, other: float):
        return self.__data / other

    def truediv_int(self, other: int):
        return self.__data / other

    def __floordiv__(self, other):
        if isinstance(other, np.float):
            v = self.floordiv_float(other)
        elif isinstance(other, MVector3D):
            v = self.floordiv_MVector3D(other)
        elif isinstance(other, np.int):
            v = self.floordiv_int(other)
        else:
            v = self.data() // other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def floordiv_MVector3D(self, other):
        return self.__data // other.__data

    def floordiv_float(self, other: float):
        return self.__data // other

    def floordiv_int(self, other: int):
        return self.__data // other

    def __mod__(self, other):
        if isinstance(other, np.float):
            v = self.mod_float(other)
        elif isinstance(other, MVector3D):
            v = self.mod_MVector3D(other)
        elif isinstance(other, np.int):
            v = self.mod_int(other)
        else:
            v = self.data() % other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def mod_MVector3D(self, other):
        return self.__data % other.__data

    def mod_float(self, other: float):
        return self.__data % other

    def mod_int(self, other: int):
        return self.__data % other

    def __lshift__(self, other):
        if isinstance(other, MVector3D):
            v = self.data() << other.data()
        else:
            v = self.data() << other
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __rshift__(self, other):
        if isinstance(other, MVector3D):
            v = self.data() >> other.data()
        else:
            v = self.data() >> other
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __and__(self, other):
        v = self.data() & other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __dataor__(self, other):
        v = self.data() ^ other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __or__(self, other):
        v = self.data() | other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __neg__(self):
        return self.__class__(-self.x(), -self.y(), -self.z())

    def __pos__(self):
        return self.__class__(+self.x(), +self.y(), +self.z())

    def x(self):
        return self.__data[0]

    def y(self):
        return self.__data[1]

    def z(self):
        return self.__data[2]
    
    def setX(self, x):
        self.__data[0] = x

    def setY(self, y):
        self.__data[1] = y

    def setZ(self, z):
        self.__data[2] = z


def crossProduct_MVector3D(v1, v2):
    crossv = np.cross(v1.data(), v2.data())
    return MVector3D(crossv[0], crossv[1], crossv[2])


def dotProduct_MVector3D(v1, v2):
    return np.dot(v1.data(), v2.data())


class MVector4D:

    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        if isinstance(x, float):
            self.__data = np.array([x, y, z, w], dtype=np.float64)
        elif isinstance(x, MVector4D):
            # クラスの場合
            self.__data = np.array([x.x(), x.y(), x.z(), x.w()], dtype=np.float64)
        elif isinstance(x, np.ndarray):
            # 行列そのものの場合
            self.__data = np.array([x[0], x[1], x[2], x[3]], dtype=np.float64)
        else:
            self.__data = np.array([x, y, z, w], dtype=np.float64)
    
    def length(self):
        return np.linalg.norm(self.data(), ord=2)

    def lengthSquared(self):
        return np.linalg.norm(self.data(), ord=2)**2

    def normalized(self):
        l2 = np.linalg.norm(self.data(), ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        normv = self.data() / l2
        return MVector4D(normv[0], normv[1], normv[2], normv[3])

    def normalize(self):
        l2 = np.linalg.norm(self.data(), ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        normv = self.data() / l2
        self.__data = normv

    def toVector3D(self):
        return MVector3D(self.__data[0], self.__data[1], self.__data[2])

    def is_almost_null(self):
        return (is_almost_null(self.__data[0]) and is_almost_null(self.__data[1]) and is_almost_null(self.__data[2]) and is_almost_null(self.__data[3]))
                   
    def effective(self):
        self.__data[np.isnan(self.data())] = 0
        self.__data[np.isinf(self.data())] = 0
                                
    @classmethod
    def dotProduct(cls, v1, v2):
        return dotProduct_MVector4D(v1, v2)
    
    def data(self):
        return self.__data

    def __str__(self):
        return "MVector4D({0}, {1}, {2}, {3})".format(self.__data[0], self.__data[1], self.__data[2], self.__data[3])

    def __lt__(self, other):
        return self.data().less(other.data())

    def __le__(self, other):
        return self.data().less_equal(other.data())

    def __eq__(self, other):
        d1 = self.data()
        d2 = other.data()
        return d1[0] == d2[0] and d1[1] == d2[1] and d1[2] == d2[2] and d1[3] == d2[3]

    def __ne__(self, other):
        d1 = self.data()
        d2 = other.data()
        return d1[0] != d2[0] or d1[1] != d2[1] or d1[2] != d2[2] or d1[3] != d2[3]

    def __eq__(self, other):
        return self.data().equal(other.data())

    def __ne__(self, other):
        return self.data().not_equal(other.data())

    def __gt__(self, other):
        return self.data().greater(other.data())

    def __ge__(self, other):
        return self.data().greater_equal(other.data())

    def __add__(self, other):
        if isinstance(other, np.float):
            v = self.add_float(other)
        elif isinstance(other, MVector4D):
            v = self.add_MVector4D(other)
        elif isinstance(other, np.int):
            v = self.add_int(other)
        else:
            v = self.data() + other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def add_MVector4D(self, other):
        return self.__data + other.__data

    def add_float(self, other: float):
        return self.__data + other

    def add_int(self, other: float):
        return self.__data + other

    def __sub__(self, other):
        if isinstance(other, np.float):
            v = self.sub_float(other)
        elif isinstance(other, MVector4D):
            v = self.sub_MVector4D(other)
        elif isinstance(other, np.int):
            v = self.sub_int(other)
        else:
            v = self.data() - other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def sub_MVector4D(self, other):
        return self.__data - other.__data

    def sub_float(self, other: float):
        return self.__data - other

    def sub_int(self, other: int):
        return self.__data - other

    def __mul__(self, other):
        if isinstance(other, np.float):
            v = self.mul_float(other)
        elif isinstance(other, MVector4D):
            v = self.mul_MVector4D(other)
        elif isinstance(other, np.int):
            v = self.mul_int(other)
        else:
            v = self.data() * other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def mul_MVector4D(self, other):
        return self.__data * other.__data

    def mul_float(self, other: float):
        return self.__data * other

    def mul_int(self, other: int):
        return self.__data * other

    def __truediv__(self, other):
        if isinstance(other, np.float):
            v = self.truediv_float(other)
        elif isinstance(other, MVector4D):
            v = self.truediv_MVector4D(other)
        elif isinstance(other, np.int):
            v = self.truediv_int(other)
        else:
            v = self.data() / other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def truediv_MVector4D(self, other):
        return self.__data / other.__data

    def truediv_float(self, other: float):
        return self.__data / other

    def truediv_int(self, other: int):
        return self.__data / other

    def __floordiv__(self, other):
        if isinstance(other, np.float):
            v = self.floordiv_float(other)
        elif isinstance(other, MVector4D):
            v = self.floordiv_MVector4D(other)
        elif isinstance(other, np.int):
            v = self.floordiv_int(other)
        else:
            v = self.data() // other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def floordiv_MVector4D(self, other):
        return self.__data // other.__data

    def floordiv_float(self, other: float):
        return self.__data // other

    def floordiv_int(self, other: int):
        return self.__data // other

    def __mod__(self, other):
        if isinstance(other, np.float):
            v = self.mod_float(other)
        elif isinstance(other, MVector4D):
            v = self.mod_MVector4D(other)
        elif isinstance(other, np.int):
            v = self.mod_int(other)
        else:
            v = self.data() % other
        v2 = self.__class__(v)
        v2.effective()
        return v2
    
    def mod_MVector4D(self, other):
        return self.__data % other.__data

    def mod_float(self, other: float):
        return self.__data % other

    def mod_int(self, other: int):
        return self.__data % other

    def __lshift__(self, other):
        if isinstance(other, MVector4D):
            v = self.data() << other.data()
        else:
            v = self.data() << other
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __rshift__(self, other):
        if isinstance(other, MVector4D):
            v = self.data() >> other.data()
        else:
            v = self.data() >> other
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __and__(self, other):
        v = self.data() & other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __dataor__(self, other):
        v = self.data() ^ other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __or__(self, other):
        v = self.data() | other.data()
        v2 = self.__class__(v)
        v2.effective()
        return v2

    def __neg__(self):
        return self.__class__(-self.x(), -self.y(), -self.z(), -self.w())

    def __pos__(self):
        return self.__class__(+self.x(), +self.y(), +self.z(), +self.w())

    def x(self):
        return self.__data[0]

    def y(self):
        return self.__data[1]

    def z(self):
        return self.__data[2]
    
    def w(self):
        return self.__data[3]
    
    def setX(self, x):
        self.__data[0] = x

    def setY(self, y):
        self.__data[1] = y

    def setZ(self, z):
        self.__data[2] = z

    def setW(self, w):
        self.__data[3] = w


def dotProduct_MVector4D(v1, v2):
    return np.dot(v1.data(), v2.data())


class MQuaternion:

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        if isinstance(w, float):
            self.__data = np.array([w, x, y, z], dtype=np.float64)
        elif isinstance(w, MQuaternion):
            # クラスの場合
            self.__data = np.array([w.data().components.w, w.data().components.x, w.data().components.y, w.data().components.z], dtype=np.float64)
        elif isinstance(w, np.quaternion):
            # quaternionの場合
            self.__data = w.components
        elif isinstance(w, np.ndarray):
            # arrayそのものの場合
            self.__data = np.array([w[0], w[1], w[2], w[3]], dtype=np.float64)
        else:
            self.__data = np.array([w, x, y, z], dtype=np.float64)

    def copy(self):
        return MQuaternion(self.scalar(), self.x(), self.y(), self.z())
    
    def __str__(self):
        return "MQuaternion({0}, {1}, {2}, {3})".format(self.scalar(), self.x(), self.y(), self.z())

    def inverted(self):
        v = self.data().inverse()
        return self.__class__(v.w, v.x, v.y, v.z)

    def length(self):
        return self.data().abs()

    def lengthSquared(self):
        return self.data().abs()**2

    def normalized(self):
        self.effective()
        v = self.data().normalized()
        return MQuaternion(v.w, v.x, v.y, v.z)

    def normalize(self):
        self.__data = self.data().normalized().components

    def effective(self):
        self.data().components[np.isnan(self.data().components)] = 0
        self.data().components[np.isinf(self.data().components)] = 0
        # Scalarは1がデフォルトとなる
        self.setScalar(1 if self.scalar() == 0 else self.scalar())

    def toMatrix4x4(self):
        mat = MMatrix4x4()
        m = mat.data()

        # q(w,x,y,z)から(x,y,z,w)に並べ替え.
        q2 = np.array([self.data().x, self.data().y, self.data().z, self.data().w], dtype=np.float64)

        m[0, 0] = q2[3] * q2[3] + q2[0] * q2[0] - q2[1] * q2[1] - q2[2] * q2[2]
        m[0, 1] = 2.0 * q2[0] * q2[1] - 2.0 * q2[3] * q2[2]
        m[0, 2] = 2.0 * q2[0] * q2[2] + 2.0 * q2[3] * q2[1]
        m[0, 3] = 0.0

        m[1, 0] = 2.0 * q2[0] * q2[1] + 2.0 * q2[3] * q2[2]
        m[1, 1] = q2[3] * q2[3] - q2[0] * q2[0] + q2[1] * q2[1] - q2[2] * q2[2]
        m[1, 2] = 2.0 * q2[1] * q2[2] - 2.0 * q2[3] * q2[0]
        m[1, 3] = 0.0

        m[2, 0] = 2.0 * q2[0] * q2[2] - 2.0 * q2[3] * q2[1]
        m[2, 1] = 2.0 * q2[1] * q2[2] + 2.0 * q2[3] * q2[0]
        m[2, 2] = q2[3] * q2[3] - q2[0] * q2[0] - q2[1] * q2[1] + q2[2] * q2[2]
        m[2, 3] = 0.0

        m[3, 0] = 0.0
        m[3, 1] = 0.0
        m[3, 2] = 0.0
        m[3, 3] = q2[3] * q2[3] + q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2]

        m /= m[3, 3]
        m[3, 3] = 1.0

        return mat
    
    def toVector4D(self):
        return MVector4D(self.data().x, self.data().y, self.data().z, self.data().w)

    def toEulerAngles4MMD(self):
        # MMDの表記に合わせたオイラー角
        euler = self.toEulerAngles()

        return MVector3D(euler.x(), -euler.y(), -euler.z())

    # http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q37
    def toEulerAngles(self):
        xp = self.data().x
        yp = self.data().y
        zp = self.data().z
        wp = self.data().w

        xx = xp * xp
        xy = xp * yp
        xz = xp * zp
        xw = xp * wp
        yy = yp * yp
        yz = yp * zp
        yw = yp * wp
        zz = zp * zp
        zw = zp * wp
        lengthSquared = xx + yy + zz + wp * wp

        if not is_almost_null(lengthSquared - 1.0) and not is_almost_null(lengthSquared):
            xx /= lengthSquared
            xy /= lengthSquared  # same as (xp / length) * (yp / length)
            xz /= lengthSquared
            xw /= lengthSquared
            yy /= lengthSquared
            yz /= lengthSquared
            yw /= lengthSquared
            zz /= lengthSquared
            zw /= lengthSquared

        pitch = asin(max(-1, min(1, -2.0 * (yz - xw))))
        yaw = 0
        roll = 0
        
        if pitch < (pi / 2):
            if pitch > -(pi / 2):
                yaw = atan2(2.0 * (xz + yw), 1.0 - 2.0 * (xx + yy))
                roll = atan2(2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz))
            else:
                # not a unique solution
                roll = 0.0
                yaw = -atan2(-2.0 * (xy - zw), 1.0 - 2.0 * (yy + zz))
        else:
            # not a unique solution
            roll = 0.0
            yaw = atan2(-2.0 * (xy - zw), 1.0 - 2.0 * (yy + zz))

        return MVector3D(degrees(pitch), degrees(yaw), degrees(roll))
    
    # 角度に変換
    def toDegree(self):
        return degrees(2 * acos(min(1, max(-1, self.scalar()))))

    # 自分ともうひとつの値vとのtheta（変位量）を返す
    def calcTheata(self, v):
        return (1 - MQuaternion.dotProduct(self.normalized(), v.normalized()))
        # dot = MQuaternion.dotProduct(self.normalized(), v.normalized())
        # angle = acos(min(1, max(-1, dot)))
        # sinOfAngle = sin(angle)
        # return sinOfAngle

    @classmethod
    def dotProduct(cls, v1, v2):
        return dotProduct_MQuaternion(v1, v2)
    
    @classmethod
    def fromAxisAndAngle(cls, vec3, angle):
        return fromAxisAndAngle(vec3, angle)

    @classmethod
    def fromAxisAndQuaternion(cls, vec3, qq):
        return fromAxisAndQuaternion(vec3, qq)

    @classmethod
    def fromDirection(cls, direction, up):
        return fromDirection(direction, up)
    
    @classmethod
    def fromAxes(cls, xAxis, yAxis, zAxis):
        return fromAxes(xAxis, yAxis, zAxis)
        
    @classmethod
    def fromRotationMatrix(cls, rot3x3):
        return fromRotationMatrix(rot3x3)

    @classmethod
    def rotationTo(cls, fromv, tov):
        return rotationTo(fromv, tov)

    @classmethod
    def fromEulerAngles(cls, pitch, yaw, roll):
        return fromEulerAngles(pitch, yaw, roll)

    @classmethod
    def nlerp(cls, q1, q2, t):
        return nlerp(q1, q2, t)

    @classmethod
    def slerp(cls, q1, q2, t):
        return slerp(q1, q2, t)

    def x(self):
        return self.data().x

    def y(self):
        return self.data().y

    def z(self):
        return self.data().z

    def scalar(self):
        return self.data().w

    def vector(self):
        return MVector3D(self.data().x, self.data().y, self.data().z)

    def setX(self, x):
        self.__data[1] = x

    def setY(self, y):
        self.__data[2] = y

    def setZ(self, z):
        self.__data[3] = z

    def setScalar(self, w):
        self.__data[0] = w
        
    def data(self):
        return np.quaternion(self.__data[0], self.__data[1], self.__data[2], self.__data[3])

    def __lt__(self, other):
        return self.data().less(other.data())

    def __le__(self, other):
        return self.data().less_equal(other.data())

    def __eq__(self, other):
        return self.data().equal(other.data())

    def __ne__(self, other):
        return self.data().not_equal(other.data())

    def __gt__(self, other):
        return self.data().greater(other.data())

    def __ge__(self, other):
        return self.data().greater_equal(other.data())

    def __add__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() + other.data()
        else:
            v = self.data() + other
        return self.__class__(v.w, v.x, v.y, v.z)

    def __sub__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() - other.data()
        else:
            v = self.data() - other
        return self.__class__(v.w, v.x, v.y, v.z)

    def __mul__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() * other.data()
            return self.__class__(v)
        elif isinstance(other, MVector3D):
            v = self.toMatrix4x4() * other
            return v
        else:
            v = self.data() * other
            return self.__class__(v.w, v.x, v.y, v.z)

    def __truediv__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() / other.data()
        else:
            v = self.data() / other
        return self.__class__(v.w, v.x, v.y, v.z)

    def __floordiv__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() // other.data()
        else:
            v = self.data() // other
        return self.__class__(v.w, v.x, v.y, v.z)

    def __mod__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() % other.data()
        else:
            v = self.data() % other
        return self.__class__(v.w, v.x, v.y, v.z)

    def __lshift__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() << other.data()
        else:
            v = self.data() << other
        return self.__class__(v.w, v.x, v.y, v.z)

    def __rshift__(self, other):
        if isinstance(other, MQuaternion):
            v = self.data() >> other.data()
        else:
            v = self.data() >> other
        return self.__class__(v.w, v.x, v.y, v.z)

    def __and__(self, other):
        v = self.data() & other.data()
        return self.__class__(v.w, v.x, v.y, v.z)

    def __dataor__(self, other):
        v = self.data() ^ other.data()
        return self.__class__(v.w, v.x, v.y, v.z)

    def __or__(self, other):
        v = self.data() | other.data()
        return self.__class__(v.w, v.x, v.y, v.z)
    
    def __neg__(self):
        return self.__class__(-self.data().w, -self.data().x, -self.data().y, -self.data().z)

    def __pos__(self):
        return self.__class__(+self.data().w, +self.data().x, +self.data().y, +self.data().z)

    def __invert__(self):
        return self.__class__(~self.data().w, ~self.data().x, ~self.data().y, ~self.data().z)


def dotProduct_MQuaternion(v1: MQuaternion, v2: MQuaternion):
    return np.sum(v1.data().components * v2.data().components)

def fromAxisAndAngle(vec3, angle: float):
    x = vec3.x()
    y = vec3.y()
    z = vec3.z()
    length = sqrt(x * x + y * y + z * z)

    if not is_almost_null(length - 1.0) and not is_almost_null(length):
        x /= length
        y /= length
        z /= length

    a = radians(angle / 2.0)
    s = sin(a)
    c = cos(a)
    return MQuaternion(c, x * s, y * s, z * s).normalized()

def fromAxisAndQuaternion(vec3, qq: MQuaternion):
    qq.normalize()

    x = vec3.x()
    y = vec3.y()
    z = vec3.z()
    length = sqrt(x * x + y * y + z * z)

    if not is_almost_null(length - 1.0) and not is_almost_null(length):
        x /= length
        y /= length
        z /= length

    a = acos(min(1, max(-1, qq.scalar())))
    s = sin(a)
    c = cos(a)

    # logger.test("scalar: %s, a: %s, c: %s, degree: %s", qq.scalar(), a, c, degrees(2 * math.acos(min(1, max(-1, qq.scalar())))))

    return MQuaternion(c, x * s, y * s, z * s).normalized()

def fromDirection(direction, up):
    if direction.is_almost_null():
        return MQuaternion()

    zAxis = direction.normalized()
    xAxis = crossProduct_MVector3D(up, zAxis)
    if (is_almost_null(xAxis.lengthSquared())):
        # collinear or invalid up vector derive shortest arc to new direction
        return rotationTo(MVector3D(0.0, 0.0, 1.0), zAxis)
    
    xAxis.normalize()
    yAxis = crossProduct_MVector3D(zAxis, xAxis)
    return MQuaternion.fromAxes(xAxis, yAxis, zAxis)

def fromAxes(xAxis, yAxis, zAxis):
    return fromRotationMatrix(np.array([[xAxis.x(), yAxis.x(), zAxis.x()], [xAxis.y(), yAxis.y(), zAxis.y()], [xAxis.z(), yAxis.z(), zAxis.z()]], dtype=np.float64))

def fromRotationMatrix(rot3x3):
    scalar = 0
    axis = np.zeros(3)

    trace = rot3x3[0,0] + rot3x3[1,1] + rot3x3[2,2]
    s = 0
    i = 0
    j = 0
    k = 0

    if trace > 0.00000001:
        s = 2.0 * sqrt(trace + 1.0)
        scalar = 0.25 * s
        axis[0] = (rot3x3[2,1] - rot3x3[1,2]) / s
        axis[1] = (rot3x3[0,2] - rot3x3[2,0]) / s
        axis[2] = (rot3x3[1,0] - rot3x3[0,1]) / s
    else:
        s_next = np.array([1, 2, 0], dtype=np.int8)
        i = 0
        if rot3x3[1,1] > rot3x3[0,0]:
            i = 1
        if rot3x3[2,2] > rot3x3[i,i]:
            i = 2

        j = s_next[i]
        k = s_next[j]

        s = 2.0 * sqrt(rot3x3[i,i] - rot3x3[j,j] - rot3x3[k,k] + 1.0)
        axis[i] = 0.25 * s

        scalar = (rot3x3[k,j] - rot3x3[j,k]) / s
        axis[j] = (rot3x3[j,i] + rot3x3[i,j]) / s
        axis[k] = (rot3x3[k,i] + rot3x3[i,k]) / s

    return MQuaternion(scalar, axis[0], axis[1], axis[2])

def rotationTo(fromv, tov):
    v0 = fromv.normalized()
    v1 = tov.normalized()
    d = MVector3D.dotProduct(v0, v1) + 1.0

    # if dest vector is close to the inverse of source vector, ANY axis of rotation is valid
    if is_almost_null(d):
        axis = crossProduct_MVector3D(MVector3D(1.0, 0.0, 0.0), v0)
        if is_almost_null(axis.lengthSquared()):
            axis = crossProduct_MVector3D(MVector3D(0.0, 1.0, 0.0), v0)
        axis.normalize()
        # same as MQuaternion.fromAxisAndAngle(axis, 180.0)
        return MQuaternion(0.0, axis.x(), axis.y(), axis.z()).normalized()

    d = sqrt(2.0 * d)
    axis = crossProduct_MVector3D(v0, v1) / d
    return MQuaternion(d * 0.5, axis.x(), axis.y(), axis.z()).normalized()

def fromEulerAngles(pitch: float, yaw: float, roll: float):
    pitch = radians(pitch)
    yaw = radians(yaw)
    roll = radians(roll)

    pitch *= 0.5
    yaw *= 0.5
    roll *= 0.5

    c1 = cos(yaw)
    s1 = sin(yaw)
    c2 = cos(roll)
    s2 = sin(roll)
    c3 = cos(pitch)
    s3 = sin(pitch)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2 * c3 + s1s2 * s3
    x = c1c2 * s3 + s1s2 * c3
    y = s1 * c2 * c3 - c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3

    return MQuaternion(w, x, y, z)

def nlerp(q1: MQuaternion, q2: MQuaternion, t: float):
    # Handle the easy cases first.
    if t <= 0.0:
        return q1
    elif t >= 1.0:
        return q2
        
    # Determine the angle between the two quaternions.
    q2b = MQuaternion(q2.scalar(), q2.x(), q2.y(), q2.z())
    
    dot = dotProduct_MQuaternion(q1, q2)
    if dot < 0.0:
        q2b = -q2b
    
    # Perform the linear interpolation.
    return (q1 * (1.0 - t) + q2b * t).normalized()

def slerp(q1: MQuaternion, q2: MQuaternion, t: float):
    # Handle the easy cases first.
    if t <= 0.0:
        return q1
    elif t >= 1.0:
        return q2

    # Determine the angle between the two quaternions.
    q2b = MQuaternion(q2.scalar(), q2.x(), q2.y(), q2.z())
    dot = dotProduct_MQuaternion(q1, q2)
    
    if dot < 0.0:
        q2b = -q2b
        dot = -dot

    # Get the scale factors.  If they are too small,
    # then revert to simple linear interpolation.
    factor1 = 1.0 - t
    factor2 = t

    if (1.0 - dot) > 0.0000001:
        angle = acos(max(0, min(1, dot)))
        sinOfAngle = sin(angle)
        if sinOfAngle > 0.0000001:
            factor1 = sin((1.0 - t) * angle) / sinOfAngle
            factor2 = sin(t * angle) / sinOfAngle

    # Construct the result quaternion.
    return q1 * factor1 + q2b * factor2


class MMatrix4x4:
    
    def __init__(self, m11=1.0, m12=0.0, m13=0.0, m14=0.0, m21=0.0, m22=1.0, m23=0.0, m24=0.0, m31=0.0, m32=0.0, m33=1.0, m34=0.0, m41=0.0, m42=0.0, m43=0.0, m44=1.0):
        if isinstance(m11, float):
            self.__data = np.array([[m11, m12, m13, m14], [m21, m22, m23, m24], [m31, m32, m33, m34], [m41, m42, m43, m44]], dtype=np.float64)
        elif isinstance(m11, MMatrix4x4):
            # 行列クラスの場合
            self.__data = np.array([[m11.__data[0, 0], m11.__data[0, 1], m11.__data[0, 2], m11.__data[0, 3]], \
                                    [m11.__data[1, 0], m11.__data[1, 1], m11.__data[1, 2], m11.__data[1, 3]], \
                                    [m11.__data[2, 0], m11.__data[2, 1], m11.__data[2, 2], m11.__data[2, 3]], \
                                    [m11.__data[3, 0], m11.__data[3, 1], m11.__data[3, 2], m11.__data[3, 3]]], dtype=np.float64)
        elif isinstance(m11, np.ndarray):
            # 行列そのものの場合
            self.__data = np.array([[m11[0, 0], m11[0, 1], m11[0, 2], m11[0, 3]], [m11[1, 0], m11[1, 1], m11[1, 2], m11[1, 3]], \
                                    [m11[2, 0], m11[2, 1], m11[2, 2], m11[2, 3]], [m11[3, 0], m11[3, 1], m11[3, 2], m11[3, 3]]], dtype=np.float64)
        else:
            # べた値の場合
            self.__data = np.array([[m11, m12, m13, m14], [m21, m22, m23, m24], [m31, m32, m33, m34], [m41, m42, m43, m44]], dtype=np.float64)

    def copy(self):
        return MMatrix4x4(self.data())
    
    def data(self):
        return self.__data

    # 逆行列
    def inverted(self):
        return MMatrix4x4(np.linalg.inv(self.data()))

    # 回転行列
    def rotate(self, qq):
        self.__data = self.data().dot(qq.toMatrix4x4().data())

    # 平行移動行列
    def translate(self, vec3):
        vec_mat = np.array([[vec3.x(), vec3.y(), vec3.z()], 
                            [vec3.x(), vec3.y(), vec3.z()], 
                            [vec3.x(), vec3.y(), vec3.z()], 
                            [vec3.x(), vec3.y(), vec3.z()]], dtype=np.float64)
        data_mat = self.__data[:, :3] * vec_mat
        self.__data[:, 3] += np.sum(data_mat, axis=1)

    # 縮尺行列
    def scale(self, scale):
        vec3 = MVector3D(scale, scale, scale)
        vec_mat = np.array([[vec3.x(), vec3.y(), vec3.z()], 
                            [vec3.x(), vec3.y(), vec3.z()], 
                            [vec3.x(), vec3.y(), vec3.z()], 
                            [vec3.x(), vec3.y(), vec3.z()]], dtype=np.float64)
        self.__data[:, :3] *= vec_mat
        
    # 単位行列
    def setToIdentity(self):
        self.__data = np.eye(4, dtype=np.float64)
    
    def lookAt(self, eye, center, up):
        forward = center - eye
        if forward.is_almost_null():
            # ほぼ0の場合終了
            return
        
        forward.normalize()
        side = crossProduct_MVector3D(forward, up).normalized()
        upVector = crossProduct_MVector3D(side, forward)

        m = MMatrix4x4()
        m.__data[0, :-1] = side.data()
        m.__data[1, :-1] = upVector.data()
        m.__data[2, :-1] = -forward.data()
        m.__data[-1, -1] = 1.0

        self *= m
        self.translate(-eye)
    
    def perspective(self, verticalAngle: float, aspectRatio: float, nearPlane: float, farPlane: float):
        if nearPlane == farPlane or aspectRatio == 0:
            return

        rad = radians(verticalAngle / 2)
        sine = sin(rad)

        if sine == 0:
            return
        
        cotan = cos(rad) / sine
        clip = farPlane - nearPlane

        m = MMatrix4x4()
        m.__data[0, 0] = cotan / aspectRatio
        m.__data[1, 1] = cotan
        m.__data[2, 2] = -(nearPlane + farPlane) / clip
        m.__data[2, 3] = -(2 * nearPlane * farPlane) / clip
        m.__data[3, 2] = -1

        self *= m
    
    def mapVector(self, vector):
        vec_mat = np.array([vector.x(), vector.y(), vector.z()])
        xyz = np.sum(vec_mat * self.__data[:3, :3], axis=1)

        return MVector3D(xyz[0], xyz[1], xyz[2])
    
    def toQuaternion(self):
        a = np.array([[self.__data[0, 0], self.__data[0, 1], self.__data[0, 2], self.__data[0, 3]],
                      [self.__data[1, 0], self.__data[1, 1], self.__data[1, 2], self.__data[1, 3]],
                      [self.__data[2, 0], self.__data[2, 1], self.__data[2, 2], self.__data[2, 3]],
                      [self.__data[3, 0], self.__data[3, 1], self.__data[3, 2], self.__data[3, 3]]], dtype=np.float64)
        
        q = MQuaternion()

        # I removed + 1
        trace = a[0,0] + a[1,1] + a[2,2]
        # I changed M_EPSILON to 0
        if trace > 0:
            s = 0.5 / sqrt(trace + 1)
            q.setScalar(0.25 / s)
            q.setX((a[2,1] - a[1,2]) * s)
            q.setY((a[0,2] - a[2,0]) * s)
            q.setZ((a[1,0] - a[0,1]) * s)
        else:
            if a[0,0] > a[1,1] and a[0,0] > a[2,2]:
                s = 2 * sqrt(1 + a[0,0] - a[1,1] - a[2,2])
                q.setScalar((a[2,1] - a[1,2]) / s)
                q.setX(0.25 * s)
                q.setY((a[0,1] + a[1,0]) / s)
                q.setZ((a[0,2] + a[2,0]) / s)
            elif a[1,1] > a[2,2]:
                s = 2 * sqrt(1 + a[1,1] - a[0,0] - a[2,2])
                q.setScalar((a[0,2] - a[2,0]) / s)
                q.setX((a[0,1] + a[1,0]) / s)
                q.setY(0.25 * s)
                q.setZ((a[1,2] + a[2,1]) / s)
            else:
                s = 2 * sqrt(1 + a[2,2] - a[0,0] - a[1,1])
                q.setScalar((a[1,0] - a[0,1]) / s)
                q.setX((a[0,2] + a[2,0]) / s)
                q.setY((a[1,2] + a[2,1]) / s)
                q.setZ(0.25 * s)

        return q

    def __str__(self):
        return "MMatrix4x4({0})".format(self.data())

    def __lt__(self, other):
        return np.all(np.less(self.data(), other.data()))

    def __le__(self, other):
        return np.all(np.less_equal(self.data(), other.data()))

    def __eq__(self, other):
        return np.all(np.equal(self.data(), other.data()))

    def __ne__(self, other):
        return np.any(np.not_equal(self.data(), other.data()))

    def __gt__(self, other):
        return np.all(np.greater(self.data(), other.data()))

    def __ge__(self, other):
        return np.all(np.greater_equal(self.data(), other.data()))

    def __add__(self, other):
        if isinstance(other, np.float):
            v = self.add_float(other)
        elif isinstance(other, MMatrix4x4):
            v = self.add_MMatrix4x4(other)
        elif isinstance(other, np.int):
            v = self.add_int(other)
        else:
            v = self.data() + other
        v2 = self.__class__(v)
        return v2
    
    def add_MMatrix4x4(self, other):
        return self.__data + other.__data

    def add_float(self, other: float):
        return self.__data + other

    def add_int(self, other: int):
        return self.__data + other

    def __sub__(self, other):
        if isinstance(other, np.float):
            v = self.sub_float(other)
        elif isinstance(other, MMatrix4x4):
            v = self.sub_MMatrix4x4(other)
        elif isinstance(other, np.int):
            v = self.sub_int(other)
        else:
            v = self.data() - other
        v2 = self.__class__(v)
        return v2
    
    def sub_MMatrix4x4(self, other):
        return self.__data - other.__data

    def sub_float(self, other: float):
        return self.__data - other

    def sub_int(self, other: int):
        return self.__data - other

    def __mul__(self, other):
        if isinstance(other, np.float):
            v = self.mul_float(other)
        elif isinstance(other, MMatrix4x4):
            v = self.mul_MMatrix4x4(other)
        elif isinstance(other, MVector3D):
            return self.mul_MVector3D(other)
        elif isinstance(other, MVector4D):
            return self.mul_MVector4D(other)
        elif isinstance(other, np.int):
            v = self.mul_int(other)
        else:
            v = self.data() * other
        v2 = self.__class__(v)
        return v2
    
    def mul_MVector3D(self, other):
        vec_mat = np.array([[other.x(), other.y(), other.z()], 
                            [other.x(), other.y(), other.z()], 
                            [other.x(), other.y(), other.z()], 
                            [other.x(), other.y(), other.z()]], dtype=np.float64)
        data_sum = np.sum(vec_mat * self.__data[:, :3], axis=1) + self.__data[:, 3]

        x = data_sum[0]
        y = data_sum[1]
        z = data_sum[2]
        w = data_sum[3]

        if w == 1.0:
            return MVector3D(x, y, z)
        elif w == 0.0:
            return MVector3D()
        else:
            return MVector3D(x / w, y / w, z / w)

    def mul_MVector4D(self, other):
        vec_mat = np.array([[other.x(), other.y(), other.z(), other.w()], 
                            [other.x(), other.y(), other.z(), other.w()], 
                            [other.x(), other.y(), other.z(), other.w()], 
                            [other.x(), other.y(), other.z(), other.w()]], dtype=np.float64)
        data_sum = np.sum(vec_mat * self.__data, axis=1)

        x = data_sum[0]
        y = data_sum[1]
        z = data_sum[2]
        w = data_sum[3]

        return MVector4D(x, y, z, w)

    def mul_MMatrix4x4(self, other):
        return np.dot(self.data(), other.data())

    def mul_float(self, other: float):
        return self.__data * other

    def mul_int(self, other: int):
        return self.__data * other

    def __iadd__(self, other):
        self.__data = self.data() + other.data().T
        return self

    def __isub__(self, other):
        self.__data = self.data() + other.data().T
        return self

    def __imul__(self, other):
        self.__data = np.dot(self.data(), other.data())

        return self

    def __itruediv__(self, other):
        self.__data = self.data() / other.data().T
        return self


def is_almost_null(v):
    return abs(v) < 0.0000001


def get_effective_value(v):
    if math.isnan(v):
        return 0
    
    if math.isinf(v):
        return 0
    
    return v


def get_almost_zero_value(v):
    if get_effective_value(v) == 0:
        return 0
        
    if is_almost_null(v):
        return 0

    return v


