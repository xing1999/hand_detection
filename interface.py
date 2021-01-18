import torch
import cv2
import numpy as np

from PIL import Image, ImageDraw

from models.ssd_new_mobilenet_FFA import build_ssd
from utils.activity_classification import ClassifyActivityRulebase

from utils.classes import Box

class SSD_Interface(object):
    def __init__(
        self,
        weight_path,
        list_classes=None,
        conf_thres=0.3,
        only_output_best=True,
    ):
        input_dim = 300 # const due to model design

        if list_classes is None:
            list_classes = [None, "hand"]

        self.list_classes = list_classes
        num_classes = len(self.list_classes)
        self.num_classes = num_classes

        self.core = build_ssd("test", input_dim, num_classes, conf_thres)
        self.input_dim = input_dim

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        c = torch.load(weight_path, map_location="cpu")


        self.core.load_state_dict(c)
        self.core.to(self.device)
        self.core.eval()

        self.activity_classifier = ClassifyActivityRulebase()
        self.prev_frame_output = []

        self.only_output_best = only_output_best


    @torch.no_grad()
    def process(self, image):
        """ Main process for SSD Model

        Args:
            image (str, Image)

        Return: List[Dict]

        Example output:
           [
		{
		    "top": 462,
		    "left": 342,
		    "bottom": 498,
		    "right": 391,
		    "cls_name": "hand",
		    "confidence": 0.3468829393386841
		}
	   ]
        """
        image, metadata = self.image_loader(image)

        w, h = metadata["width"], metadata["height"]
        detections = self.core(image).data
        results = []
        for j in range(1, self.num_classes):
            dets = detections[0, j, :]

            mask = dets[:, 0].gt(0.).expand(11, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 11)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:5]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)

            json_result = [self.jsontify(b, j) for b in cls_dets]

            if self.only_output_best:
                json_result = [max(json_result, key=lambda x: Box(x).area)]
            results.extend(json_result)

        self.prev_frame_output = results

        return results

    def jsontify(self, single_box, cls_index):
        conf = float(single_box[-1])
        x1, y1, x2, y2 = np.array(single_box[:4], dtype=int).tolist()
        cls_name = self.list_classes[cls_index]

        info = {
            "top": y1,
            "left": x1,
            "bottom": y2,
            "right": x2,
            "cls_name": cls_name,
            "confidence": conf,
        }

        info["activity_type"] = self.classify_activity(info)
        return info

    def classify_activity(self, info):
        """ Classify activity based on previous frame """

        if len(self.prev_frame_output) == 0:
            prev_frame = None
        elif len(self.prev_frame_output) > 1:
            # Not implement object mapping yet...
            prev_frame = None
        else:
            prev_frame = self.prev_frame_output[0]

        return self.activity_classifier.process(prev_frame, info)


    def image_loader(self, image):
        """ Warper for image loading function """
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        h, w, c = image.shape
        input_dim = self.input_dim

        image = cv2.resize(image, (input_dim, input_dim))
        x = (image.astype(np.float32) / 255.0 - 0.5)*2
        metadata = {"width": w, "height": h}
        x = torch.tensor(x).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0)
        return x, metadata

if __name__ == "__main__":
    weight_path = "./weights/xing_weight.pth"
    model = SSD_Interface(weight_path)

    image_path = "./data/sample/input.jpg"
    output = model.process(image_path)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    print(f"Found {len(output)} boxes")
    import json
    print(json.dumps(output, indent=4))
