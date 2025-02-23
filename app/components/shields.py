import glob
import os
import time

import cv2


class GetShields():
    def __init__(self):
        self.shield_templates = self.generate_shield_templates()

    def generate_shield_templates(self):
        generated_shield_templates = []
        template_directory = os.path.abspath(os.path.join(
            __file__, "../../templates/shield_templates"))
        all_templates = [image for image in glob.glob(
            template_directory+"/*.png")]
        for template in all_templates:
            shield = (template.split(r"\shield_templates"))[
                1].split(".png")[0][1:]
            generated_shield_templates.append({
                "shield": shield,
                "gray": cv2.imread(template, 0)
            })
        return generated_shield_templates

    def process_shields_frame(self, image):
        all_results = []
        for template in self.shield_templates:
            current_template = template["gray"]
            result = cv2.matchTemplate(
                image, current_template, cv2.TM_CCOEFF_NORMED)
            all_results.append((cv2.minMaxLoc(result)[1], template["shield"]))
        if max(all_results)[0] > 0.4:
            return max(all_results)[1]
        else:
            return None

    def identify_shields(self, frame, side):
        debug_frame = frame.copy()
        all_identified_shields = []
        if side == "top":
            y_start = 451
        else:
            y_start = 760
        y_end = y_start + 45

        for agent_loadout in range(0, 5):
            # Draw rectangle for shield region
            if side == "top":
                start_point = (1567, y_start)
                end_point = (1602, y_end)
            else:
                start_point = (1565, y_start)
                end_point = (1600, y_end)
            cv2.rectangle(debug_frame, start_point, end_point, (0, 255, 0), 2)
            if side == "top":
                resized_frame = frame[y_start:y_end, 1567:1602]
            else:
                resized_frame = frame[y_start:y_end, 1565:1600]
            identified_weapon = self.process_shields_frame(resized_frame)
            all_identified_shields.append(identified_weapon)
            y_start = y_start + 45
            y_end = y_end + 45
        
        cv2.imwrite(f'debug_shields_{side}.png', debug_frame)
        return all_identified_shields

    def get_shields(self, frame):
        all_identified_shields = {}
        main_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_identified_shields["top"] = self.identify_shields(
            main_frame_gray, "top")
        all_identified_shields["bottom"] = self.identify_shields(
            main_frame_gray, "bottom")
        return all_identified_shields


if __name__ == "__main__":
    get_all_shields = GetShields()
    tab_images_directory = os.path.abspath(
        os.path.join(__file__, "../../test_images/Tab Images/"))
    for i in range(1, 7):
        start = time.time()
        image = cv2.imread('{}/{}.png'.format(tab_images_directory, i))
        print("===================Image No. {}===================".format(i))
        identified_weapons = get_all_shields.get_shields(image)
        print("Identified Weapons", identified_weapons)
        end = time.time()
        print("Time elapsed", end - start)