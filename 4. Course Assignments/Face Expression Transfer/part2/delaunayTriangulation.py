import matplotlib.pyplot as plt
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "("+str(self.x)+" , "+str(self.y)+")"
    
    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

class Triangle:
    def __init__(self, p1, p2, p3):
        sorted_points = list(sorted([p1,p2,p3], key=lambda p: (p.x, p.y)))
        # print(sorted_points)
        self.p1 = sorted_points[0]
        self.p2 = sorted_points[1]
        self.p3 = sorted_points[2]
        self.circumcenter, self.circumcenter_radius = self.calculate_circumcenter_radius()

    def calculate_circumcenter_radius(self):
        # Calculate circumcenter and circumradius of the triangle
        ax, ay, bx, by, cx, cy = self.p1.x, self.p1.y, self.p2.x, self.p2.y, self.p3.x, self.p3.y
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        circumcenter = Point(ux, uy)
        circumradius = math.sqrt((ax - ux)**2 + (ay - uy)**2)
        return circumcenter, circumradius
    
    def __hash__(self):
        return hash(((self.p1.x, self.p1.y),(self.p2.x, self.p2.y),(self.p3.x, self.p3.y)))

    def __eq__(self, other):
        if isinstance(other, Triangle):
            return self.p1.x == other.p1.x and self.p1.y == other.p1.y and self.p2.x == other.p2.x and self.p2.y == other.p2.y and self.p3.x == other.p3.x and self.p3.y == other.p3.y
        return False

    def __str__(self):
        return "("+str(self.p1)+" , "+str(self.p2)+" , "+str(self.p3)+")"

def bowyer_watson(list):
    points=[]
    for i in list:
        points.append(Point(i[0],i[1]))
    # print(points)
    # Create a super triangle that contains all points
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    p1 = Point(mid_x - 5 * delta_max, mid_y - delta_max)
    p2 = Point(mid_x, mid_y + 5 * delta_max)
    p3 = Point(mid_x + 5 * delta_max, mid_y - delta_max)

    # print(p1,p2,p3)
    super_triangle=Triangle(p1, p2, p3)

    # for point in points:
    #     plt.plot(point.x, point.y, 'ro')
    # plt.plot(p1.x, p1.y, 'x')
    # plt.plot(p2.x, p2.y, 'x')
    # plt.plot(p3.x, p3.y, 'x')
    # plt.show()

    triangles=set()
    triangles.add(super_triangle)
    # checking if a point is inside the circulcircle of a triangle
    for point in points:
        bad_triangles = []
        
        for triangle in triangles:
            if (point.x - triangle.circumcenter.x) ** 2 + (point.y - triangle.circumcenter.y) ** 2 <= triangle.circumcenter_radius ** 2:
                bad_triangles.append(triangle)

        edges = set()

        for triangle in bad_triangles:
            # for edge in [{triangle.p1, triangle.p2}, {triangle.p2, triangle.p3}, {triangle.p3, triangle.p1}]:
            for edge in [(triangle.p1, triangle.p2), (triangle.p2, triangle.p3), (triangle.p3, triangle.p1),(triangle.p2, triangle.p1),(triangle.p3, triangle.p2), (triangle.p1, triangle.p3)]:
                if edge in edges:
                    edges.remove(edge)
                else:
                    edges.add(edge)
            triangles.remove(triangle)

        for edge in edges:
            triangles.add(Triangle(point, edge[0], edge[1]))

    triangles = [t for t in triangles if not any(point in [super_triangle.p1, super_triangle.p2, super_triangle.p3] for point in [t.p1, t.p2, t.p3])]
    return triangles


def plot_delaunay(triangles):
    for triangle in triangles:
        plt.plot([triangle.p1.x, triangle.p2.x, triangle.p3.x, triangle.p1.x],
                 [triangle.p1.y, triangle.p2.y, triangle.p3.y, triangle.p1.y], 'b')

    plt.show()


def mapTriangles(Dtriangles, listA, listB):
    pointsA=[]
    pointsB=[]
    for i in listA:
        pointsA.append(Point(i[0],i[1]))
    for i in listB:
        pointsB.append(Point(i[0],i[1]))
    
    #creating mapping between points
    dicti=dict()
    for i in range(len(pointsA)):
        dicti[pointsA[i]]=pointsB[i]
        # print(pointsA[i],pointsB[i])
    
    for t in Dtriangles:
        # print(t)
        t.p1=dicti[t.p1]
        t.p2=dicti[t.p2]
        t.p3=dicti[t.p3]

    return Dtriangles