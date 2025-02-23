import glob
import os
import time

import cv2
import numpy as np


class GetLiveAgents():
    def __init__(self):
        self.agent_templates = self.generate_agent_templates()

    def generate_agent_templates(self):
        generated_agent_templates = []
        template_directory = os.path.abspath(os.path.join(
            __file__, "../../templates/agent_templates"))
        all_templates = [image for image in glob.glob(
            template_directory+"/*_icon.png")]
        for template in all_templates:
            agent = (template.split(r"\agent_templates"))[
                1].split("_icon")[0][1:]
            original = cv2.resize(cv2.imread(
                template, cv2.IMREAD_UNCHANGED), (59, 59))
            gray = cv2.resize(cv2.imread(template, 0), (59, 59))
            o_ret, original_mask = cv2.threshold(
                original[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            f_ret, original_flipped_mask = cv2.threshold(
                cv2.flip(original[:, :, 3], 1), 0, 255, cv2.THRESH_BINARY)
            flipped_gray = cv2.flip(gray, 1)
            generated_agent_templates.append({
                "agent": agent,
                "original": original,
                "gray": gray,
                "original_mask": original_mask,
                "original_flipped_mask": original_flipped_mask,
                "flipped_gray": flipped_gray
            })
        return generated_agent_templates

    def identify_agent(self, resized_frame, side):
        identified_agent = None
        for template in self.agent_templates:
            if side == "right":
                result = cv2.matchTemplate(
                    resized_frame, template["flipped_gray"], cv2.TM_CCOEFF_NORMED, None, template["original_flipped_mask"])
            else:
                result = cv2.matchTemplate(
                    resized_frame, template["gray"], cv2.TM_CCOEFF_NORMED, None, template["original_mask"])
            max_location = cv2.minMaxLoc(result)[1]
            if max_location > 0.7:
                identified_agent = template["agent"]
        return identified_agent

    def process_frame(self, screen_frame, side):
        debug_frame = screen_frame.copy()
        all_agents = []
        width = 59
        space_between_agents = 88
        if side == "left":
            x_start = 589
        else:
            x_start = 1558
        x_end = x_start + width

        for agent_place in range(0, 5):
            # Draw rectangle for agent region
            start_point = (x_start, 37)
            end_point = (x_end, 96)
            cv2.rectangle(debug_frame, start_point, end_point, (0, 255, 0), 2)
            
            cropped_agent_image = screen_frame[37:96, x_start:x_end]
            identified_agent = self.identify_agent(cropped_agent_image, side)
            all_agents.append(identified_agent)
            x_start = x_start + space_between_agents
            x_end = x_start + width
        
        cv2.imwrite(f'debug_header_agents_{side}.png', debug_frame)
        return all_agents

    def get_agents(self, main_frame):
        agents_alive = {"left": [], "right": []}
        main_frame_gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
        agents_alive["left"] = self.process_frame(main_frame_gray, "left")
        agents_alive["right"] = self.process_frame(main_frame_gray, "right")
        return agents_alive


if __name__ == "__main__":

    get_live_agents = GetLiveAgents()
    feed_images_directory = os.path.abspath(
        os.path.join(__file__, "../../test_images/Feed Images/"))
    for i in range(1, 6):
        print("=======================")
        print("FILE: ", i)
        start = time.time()
        img = cv2.imread('{}/feed{}.png'.format(feed_images_directory, i))
        agents_alive = get_live_agents.get_agents(img)
        print(agents_alive)
        end = time.time()
        print("Time elapsed", end - start)