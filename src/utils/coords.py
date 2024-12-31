import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Iterator, Tuple, Union


class StraightLine(ABC):
    """
    Abstract class to represent a straight line.
    """

    @property
    @abstractmethod
    def m(self) -> float:
        """
        Returns the slope of the line.
        """
        pass

    @property
    @abstractmethod
    def q(self) -> float:
        """
        Returns the y-intercept of the line.
        """
        pass


class Position(ABC):
    """
    Abstract class to represent a position in a 2D space.
    This class is used to represent points, nodes on the road network graph and points of interest.
    """

    @abstractmethod
    def distance_to(self, coords: "Coords") -> float:
        """
        Returns the Euclidean distance between the position and another one.
        """
        pass

    @abstractmethod
    def closest_point(self, coords: "Coords") -> "Coords":
        """
        Returns the closest point to the position from a given point.
        """
        pass

    @abstractmethod
    def get_complete_description(self) -> str:
        """
        Returns a complete description of the position.
        This is used when the user wants to know the complete information of a position.
        """
        pass


@dataclass(frozen=True)
class Coords(Position):
    """
    Class to represent a pair of coordinates (x, y) or a 2D vector.
    Instances are immutable. Math operations always return a new instance.
    """

    x: float
    "X coordinate."
    y: float
    "Y coordinate."

    ZERO: ClassVar["Coords"]
    "Zero coordinates (0, 0)."

    INF: ClassVar["Coords"]
    "Infinite coordinates."

    @property
    def coords(self) -> Tuple[float, float]:
        """
        Returns the coordinates as a tuple (x, y).
        """
        return self.x, self.y

    def closest_point(self, coords: "Coords") -> "Coords":
        return self

    def get_complete_description(self) -> str:
        return str(self)

    def distance_to(self, coords: "Coords") -> float:
        """
        Returns the Euclidean distance between the point and another one.
        """
        return float(((self.x - coords.x) ** 2 + (self.y - coords.y) ** 2) ** 0.5)

    def manhattan_distance_to(self, other: "Coords") -> float:
        """
        Returns the Manhattan distance between the point and another one.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distance_to_line(self, line: StraightLine) -> float:
        """
        Returns the distance between the point and a straight line.
        """
        if math.isinf(line.m):
            return abs(self.x - line.q)

        num = abs(line.m * self.x + line.q - self.y)
        den = (line.m**2 + 1) ** 0.5

        return float(num / den)

    def project_on(self, line: StraightLine) -> "Coords":
        """
        Returns the projection of the point on a straight line.
        """
        if math.isinf(line.m):
            return Coords(line.q, self.y)

        p_x = (self.x + line.m * self.y - line.m * line.q) / (line.m**2 + 1)
        p_y = (line.m * self.x + line.m**2 * self.y + line.q) / (line.m**2 + 1)

        return Coords(p_x, p_y)

    def dot(self, other: "Coords") -> float:
        """
        Returns the dot product of the vector with another one.
        """
        return self.x * other.x + self.y * other.y

    def cross_2d(self, other: "Coords") -> float:
        """
        Returns the 2D cross product of the vector with another one.
        """
        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        """
        Returns the length of the vector.
        """
        return self.distance_to(Coords.ZERO)

    def magnitude(self) -> float:
        """
        Returns the length of the vector.
        """
        return self.length()

    def normalized(self) -> "Coords":
        """
        Returns a new instance with the same direction but with length 1.
        """
        return self / self.length()

    def __add__(self, other: Union["Coords", float]) -> "Coords":
        if isinstance(other, Coords):
            return Coords(self.x + other.x, self.y + other.y)
        return Coords(self.x + other, self.y + other)

    def __sub__(self, other: Union["Coords", float]) -> "Coords":
        if isinstance(other, Coords):
            return Coords(self.x - other.x, self.y - other.y)
        return Coords(self.x - other, self.y - other)

    def __mul__(self, other: float) -> "Coords":
        return Coords(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> "Coords":
        return Coords(self.x / other, self.y / other)

    def __floordiv__(self, other: float) -> "Coords":
        return Coords(self.x // other, self.y // other)

    def __getitem__(self, index: int) -> float:
        """
        Returns the x coordinate if index is 0, the y coordinate otherwise.
        If index is not 0 or 1, raises an IndexError.
        """
        return self.coords[index]

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y))

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __round__(self, n: int = 0) -> "Coords":
        return Coords(round(self.x, n), round(self.y, n))

    def __repr__(self) -> str:
        return str(self)


Coords.ZERO = Coords(0, 0)
Coords.INF = Coords(math.inf, math.inf)


class LatLngReference:
    """
    Class to store the reference coordinates for the conversion between pixels and latitude and longitude.
    """

    def __init__(self, coords: Coords, lat: float, lng: float) -> None:
        """
        Initializes the reference coordinates.

        :param coords: Coordinates in pixels of the reference point.
        :param lat: Latitude of the reference point.
        :param lng: Longitude of the reference point.
        """

        self.coords = coords
        """Coordinates in pixels of the reference point."""

        self.lat = lat
        """Latitude of the reference point."""

        self.lng = lng
        """Longitude of the reference point."""


feets_per_meter = 3.280839895
"""Conversion factor from meters to feets."""

R = 6378137 * feets_per_meter
"""Earth radius in feets."""


def coords_to_latlng(latlng_reference: LatLngReference, coords: Coords) -> Coords:
    """
    Converts a pair of coordinates in pixels to a latitude and longitude pair.
    """

    diff = coords - latlng_reference.coords
    de = diff[0]
    dn = -(diff[1])

    dLat = dn / R
    dLon = de / (R * math.cos(math.pi * latlng_reference.lat / 180))

    latO = latlng_reference.lat + dLat * 180 / math.pi
    lonO = latlng_reference.lng + dLon * 180 / math.pi

    return Coords(latO, lonO)


def latlng_to_coords(reference: LatLngReference, latlng: Coords) -> Coords:
    """
    Converts a latitude and longitude pair to a pair of coordinates in pixels.
    """

    dx = latlng_distance(reference.lat, reference.lng, reference.lat, latlng.y)
    dy = latlng_distance(reference.lat, reference.lng, latlng.x, reference.lng)

    if reference.lat > latlng.x:
        dy *= -1
    if reference.lng > latlng.y:
        dx *= -1

    return Coords(reference.coords.x + dx, reference.coords.y - dy)


def latlng_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Returns the distance between two points on the Earth's surface given their latitude and longitude.
    """

    lat1 = math.radians(lat1)
    lng1 = math.radians(lng1)
    lat2 = math.radians(lat2)
    lng2 = math.radians(lng2)

    dLat = lat2 - lat1
    dLon = lng2 - lng1

    a = (
        math.sin(dLat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))
    return R * c
