from math import hypot


class Vector(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Vector({},{})".format(self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))\


    def __add__(self, other):
        x = self.x+other.x
        y = self.y+other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x*scalar, self.y*scalar)


def cartesian_product():
    """
    使用列表推导计算笛卡儿积 
    """
    colors = ["black", "white"]
    sizes = ["S", "M", "L"]
    tshirts = [(colors, size) for color in colors for size in sizes]

    print(tshirts)
    # 排序
    for color in colors:
        for size in sizes: 
            print((color, size)) 


    tshirts = [(color, size) for size in sizes for color in colors]
    print("")
    print(tshirts)



def main():
    v = Vector(x=1, y=2)
    r0 = v.__repr__()
    r1 = v.__abs__()
    r2 = v.__bool__()
    r3 = v.__add__(v)
    r4 = v.__add__(v)
    print(r0, r1, r2, r3, r4)


if __name__ == '__main__':
    # main()

    cartesian_product()
