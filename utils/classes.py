class Box(object):
    """ Warper for object detection data """
    def __init__(self, info: dict):
        self.left = info["left"]
        self.right = info["right"]
        self.top = info["top"]
        self.bottom = info["bottom"]


    def horizontal_intersection(self, other):
       return max(min(self.right, other.right) - max(self.left, other.left), 0)

    def vertical_intersection(self, other):
       return max(min(self.bottom, other.bottom) - max(self.top, other.top), 0)

    def intersection(self, other):
       h_intersect = self.horizontal_intersection(other)
       v_intersect = self.vertical_intersection(other)
       return h_intersect * v_intersect

    def iou(self, other):
       intersect = self.intersection(other)
       return intersect / (self.area + other.area - intersect)


    @property
    def height(self):
        return self.bottom - self.top

    @property
    def width(self):
        return self.right - self.left

    @property
    def area(self):
        return self.height * self.width

    @property
    def v_center(self):
        return (self.top - self.bottom) / 2

    @property
    def h_center(self):
        return (self.right  - self.left) / 2
