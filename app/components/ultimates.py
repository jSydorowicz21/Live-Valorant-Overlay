import os
import time

import cv2
import numpy as np
from easyocr import Reader


class GetUltimates():
    def __init__(self):
        self.reader = Reader(["en"])
        # pass

    def cleanup_text(self, text):
        # strip out non-ASCII text
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    def clean_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, None, fx=8, fy=8,
                           interpolation=cv2.INTER_CUBIC)
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        frame = cv2.threshold(
            frame, 240, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((4, 4), np.uint8)
        frame = cv2.dilate(frame, kernel, iterations=1)
        frame = cv2.erode(frame, kernel, iterations=1)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        return frame

    def process_frame(self, frame, side):
        # Create a copy for visualization
        debug_frame = frame.copy()
        
        all_points = []
        if side == "top":
            y_start = 451
        else:
            y_start = 760
        y_end = y_start + 45

        for agent_row in range(0, 5):
            # Draw rectangle for current region
            if side == "top":
                start_point = (1140, y_start)
                end_point = (1154, y_end)
            else:
                start_point = (1136, y_start)
                end_point = (1152, y_end)
            cv2.rectangle(debug_frame, start_point, end_point, (0, 255, 0), 2)  # Green rectangle
            
            if side == "top":
                cropped_score_image = frame[y_start:y_end, 1140:1158]
            else:
                cropped_score_image = frame[y_start:y_end, 1136:1160]
            points = self.reader.readtext(self.clean_frame(
                cropped_score_image), allowlist='0123456789')
            
            if not points:
                points = "READY"
            else:
                points = {"number": points[0][1], "confidence": points[0][2]}
                if points["confidence"] < .6:
                    print("Low confidence", points["confidence"], points["number"])
                    points = "READY"
            all_points.append(points)
            y_start = y_start + 45
            y_end = y_start + 45

        # Save debug image
        cv2.imwrite(f'debug_ultimates_{side}.png', debug_frame)
        return all_points

    def get_ultimate_points(self, frame):
        all_ultimates = {"top": [], "bottom": []}
        all_ultimates["top"] = self.process_frame(frame, "top")
        all_ultimates["bottom"] = self.process_frame(frame,"bottom")
        return all_ultimates


if __name__ == "__main__":
    get_all_ultimates = GetUltimates()
    tab_images_directory = os.path.abspath(
        os.path.join(__file__, "../../test_images/Tab Images/"))
    for i in range(1, 7):
        start = time.time()
        image = cv2.imread('{}/{}.png'.format(tab_images_directory, i))
        print("===================Image No. {}===================".format(i))
        all_points = get_all_ultimates.get_ultimate_points(image)
        print("all_points", all_points)
        end = time.time()
        print("Time elapsed", end - start)
