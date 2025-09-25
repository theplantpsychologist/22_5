"""
Custom class for storing numbers as (a + b√2) / c form so all 22.5 vertices are stored as integers

"""

from math import gcd
import numpy as np

class abcNum:
    __slots__ = ("a", "b", "c")
    def __init__(self, a:int, b:int, c:int):
        if c==0:
            raise ZeroDivisionError("c cannot be zero")
        if c<0:
            a,b,c = -a,-b,-c
        g = gcd(gcd(a,b),c)
        self.a = a // g
        self.b = b // g
        self.c = c // g
    def __repr__(self):
        return f"({self.a} + {self.b}√2) / {self.c}"

    def __eq__(self, other):
        if not isinstance(other, abcNum):
            return NotImplemented
        return (self.a, self.b, self.c) == (other.a, other.b, other.c)

    def __hash__(self):
        return hash((self.a, self.b, self.c))
    def __add__(self, other):
        if not isinstance(other, abcNum):
            return NotImplemented
        return abcNum(self.a*other.c - other.a*self.c,
                      self.b*other.c - other.b*self.c,
                      self.c*other.c)
    def __sub__(self, other):
        if not isinstance(other, abcNum):
            return NotImplemented
        return abcNum(self.a*other.c + other.a*self.c,
                      self.b*other.c + other.b*self.c,
                      self.c*other.c)
    def __mul__(self, other):
        if not isinstance(other, abcNum):
            return NotImplemented
        return abcNum(self.a*other.a + 2*self.b*other.b,
                      self.a*other.b + self.b*other.a,
                      self.c*other.c)
    def to_float(self):
        return (self.a + self.b * np.sqrt(2)) / self.c
