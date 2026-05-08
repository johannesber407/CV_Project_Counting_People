import cv2
import numpy as np
from dataclasses import dataclass, field


class Circle:
    def __init__(self, center=(0., 0.), radius=0.0):
        """
        Construct a circle from a center point and a radius.

        :param center: The center point of the circle.
        :param radius: The radius of the circle.
        """
        self._center = center
        self._radius = radius

    @classmethod
    def from_points(cls, point1, point2, point3):
        """
        Constructs a circle from three points on the circle.

        :param point1: First point on the circle.
        :param point2: Second point on the circle.
        :param point3: Third point on the circle.
        :return:
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        point3 = np.asarray(point3)

        def homogeneous(x):
            return np.append(x, [1], axis=0)

        def hnormalized(x):
            return x[:-1] / x[-1]

        def center_line(p1, p2):
            m1 = 0.5 * (p1 + p2)
            q1 = (m1[0] + p2[1] - p1[1],
                  m1[1] - p2[0] + p1[0])
            return np.cross(homogeneous(m1), homogeneous(q1))

        # Compute homogeneous center lines.
        line_1 = center_line(point1, point2)
        line_2 = center_line(point2, point3)

        # Compute homogenous center point.
        center_h = np.cross(line_1, line_2)

        # Co-linear points will result in a center at infinity,
        # so return invalid circle in that case.
        if center_h[-1] == 0:
            return None

        # Construct circle.
        center = hnormalized(center_h)
        radius = np.linalg.norm(point1 - center)
        return cls(center, radius)

    @property
    def center(self):
        """
        The center point of the circle.
        :return: The center point.
        """
        return self._center

    @property
    def radius(self):
        """
        The circle radius.
        :return: The radius.
        """
        return self._radius

    def distances(self, points):
        """
        The distance from the circle to each given point.
        :param points: A nx2 set of points
        :return: The distances.
        """
        points = np.asarray(points)

        if points.ndim == 1:
            points = points[np.newaxis, :]

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points.shape {points.shape} != nx2")

        return np.abs(np.linalg.norm(points - self._center, axis=1) - self._radius)

    def __str__(self):
        return f"center: {self._center}, radius: {self._radius}"


@dataclass
class CircleEstimate:
    """Datatype for circle estimate as a result from CircleEstimator"""
    circle: Circle = Circle()
    num_iterations: int = 0
    num_inliers: int = 0
    is_inlier: np.ndarray = field(default_factory=lambda: np.array(object=[], dtype=bool))


def create_1d_gaussian_kernel(sigma, radius=0):
    """
    Creates a Nx1 gaussian filter kernel.

    :param sigma: The sigma (standard deviation) parameter for the gaussian.
    :param radius: The filter radius, so that N = 2*radius + 1. If set to 0, the radius will be computed so that radius = 3.5 * sigma.
    :return: Nx1 gaussian filter kernel.
    """
    if radius <= 0:
        radius = int(np.ceil(3.5 * sigma))

    length = 2 * radius + 1
    x = np.arange(0, length) - radius
    kernel = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x * x / (2 * sigma * sigma))

    return kernel, x


def draw_circle_result(img, keypoints, circle_estimate, duration):
    # If not a result
    if not circle_estimate:
        return

    cv2.putText(img, f"circle time: {duration:.2f}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0))

    # Extract and draw circle point inliers
    inlier_pts = extract_inlier_points(circle_estimate, keypoints)
    cv2.drawKeypoints(img, inlier_pts, img, (0, 0, 255))

    # Draw estimated circle
    center = np.asarray(circle_estimate.circle.center, dtype=int)
    radius = int(circle_estimate.circle.radius)
    cv2.circle(img, np.flip(center), radius, (0, 0, 255), cv2.LINE_4, cv2.LINE_AA)


def draw_corner_result(img, keypoints, duration):
    cv2.putText(img, f"corner time: {duration:.2f}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0))
    cv2.drawKeypoints(img, keypoints, img, (0, 255, 0))


def extract_inlier_points(estimate, points):
    return points[estimate.is_inlier]


def retain_best(keypoints, num_to_keep):
    num_to_keep = np.minimum(num_to_keep, len(keypoints))
    best = np.argpartition([p.response for p in keypoints], -num_to_keep)[-num_to_keep:]
    return best