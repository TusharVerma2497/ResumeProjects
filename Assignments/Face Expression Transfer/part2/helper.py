from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def readImage(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image



def getAffineTransformSolvingSLI(source_points, target_points):
    if (source_points.shape[0] < 3 or target_points.shape[0]<3):
        raise ValueError("At least 3 pairs of corresponding points are required.")

    A = np.zeros((2 * source_points.shape[0], 6))
    b = np.zeros((2 * source_points.shape[0], 1))
    for i in range(source_points.shape[0]):
        x, y = source_points[i]
        x_prime, y_prime = target_points[i]

        A[i * 2, :] = [x, y, 1, 0, 0, 0]
        A[i * 2 + 1, :] = [0, 0, 0, x, y, 1]

        b[i * 2] = x_prime
        b[i * 2 + 1] = y_prime

    affine_parameters, _ = np.linalg.lstsq(A, b, rcond=None)[0], None

    # Reshape the affine parameters into a 2x3 transformation matrix
    affine_matrix = np.vstack([affine_parameters[:3], affine_parameters[3:]]).reshape(2,3)
    return affine_matrix



def getAffineTransformUsingCV2(source_points, target_points):
    if (source_points.shape[0] < 3 or target_points.shape[0]<3):
        raise ValueError("At least 3 pairs of corresponding points are required.")
    M=cv2.getAffineTransform(source_points, target_points)

    return M



def performTransformation(points, matrix):
    points_3d = np.hstack((points, np.ones((points.shape[0], 1))))
    return np.float32(np.matmul(points_3d,matrix.T))




def getBarycentricCoordinates(triangle, point):
    # Extract the vertices of the triangle
    A, B, C = triangle

    # Calculate the vectors from each vertex to the point
    v0 = [C[0] - A[0], C[1] - A[1]]
    v1 = [B[0] - A[0], B[1] - A[1]]
    v2 = [point[0] - A[0], point[1] - A[1]]

    # Calculate dot products and cross product
    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    # Calculate barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1.0 - u - v

    return u, v, w


# def pointInBoundary(x, y, boundary):

#     def area(x1, y1, x2, y2, x3, y3):
#         return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
#                 + x3 * (y1 - y2)) / 2.0)
    
#     x1=boundary[0][0]
#     y1=boundary[0][1]
#     x2=boundary[1][0]
#     y2=boundary[1][1]
#     x3=boundary[2][0]
#     y3=boundary[2][1]

#     A = np.round(area(x1, y1, x2, y2, x3, y3))
#     A1 = (area(x, y, x2, y2, x3, y3))
#     A2 = (area(x1, y1, x, y, x3, y3))
#     A3 = (area(x1, y1, x2, y2, x, y))
     
#     # Check if sum of A1, A2 and A3 is same as A
#     return A == np.round(A1 + A2 + A3)

def areaTriangle(x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)

def pointInBoundary(x, y, boundary, A):
    
    x1=boundary[0][0]
    y1=boundary[0][1]
    x2=boundary[1][0]
    y2=boundary[1][1]
    x3=boundary[2][0]
    y3=boundary[2][1]

    
    A1 = (areaTriangle(x, y, x2, y2, x3, y3))
    A2 = (areaTriangle(x1, y1, x, y, x3, y3))
    A3 = (areaTriangle(x1, y1, x2, y2, x, y))
     
    # Check if sum of A1, A2 and A3 is same as A
    return A == np.round(A1 + A2 + A3)


def getPointUsingBarycentric(triangle, barycentric):
    # Extract the vertices of the triangle
    A, B, C = triangle

    # Extract the barycentric coordinates
    w, v, u = barycentric
    # u, v, w = barycentric

    # Calculate the Cartesian coordinates of the point
    x = u * A[0] + v * B[0] + w * C[0]
    y = u * A[1] + v * B[1] + w * C[1]

    return x, y





def RGBtoLAB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

def LABtoRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_Lab2RGB)




def showImage(image, name='img', cmap="viridis",):
    plt.axis('off')
    plt.title(name)
    plt.imshow(image, cmap)
    plt.show()