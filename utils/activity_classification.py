from utils.classes import Box

from typing import Optional


class ClassifyActivityRulebase(object):
    def __init__(self, iou_thres=0.7, h_thres=0.4, default="wave"):
        self.iou_thres = iou_thres
        self.h_thres = h_thres
        self.default = default

    def process(self, prev: Optional[dict,None], current: dict) -> str:
        if prev is not None:
            prev = Box(prev)
        current = Box(current)
        return self._process(prev, current)


    def _process(self, prev_frame: Box, item: Box) -> str:
        if prev_frame is None:
            return "idle"

        # The object likely still in same box
        if item.iou(prev_frame) > self.iou_thres:
            return "idle"

        # Object only move up
        h_intersect = item.horizontal_intersection(prev_frame)
        is_move_up = item.v_center > prev_frame.v_center
        if h_intersect > self.h_thres and is_move_up:
            return "jump"

        # Other
        return self.default
