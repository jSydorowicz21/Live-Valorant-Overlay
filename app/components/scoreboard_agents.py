import glob
import os
import time

import cv2


class GetScoreBoardAgents():
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
            original = cv2.resize(cv2.imread(template), (44, 44))  # Updated from 33,33 for 1440p
            generated_agent_templates.append({
                "agent": agent,
                "original": original
            })
        return generated_agent_templates

    def identify_agent(self, resized_frame):
        all_results = []
        for template in self.agent_templates:
            current_template = template["original"]
            result = cv2.matchTemplate(
                resized_frame, current_template, cv2.TM_CCOEFF_NORMED)
            all_results.append((cv2.minMaxLoc(result)[1], template["agent"]))
        return max(all_results)[1]

    def process_frame(self, screen_frame, side):
        all_agents = []
        if side == "top":
            y_start = 451  # Updated from 338 for 1440p (338 * 1.333)
        else:
            y_start = 760  # Updated from 570 for 1440p (570 * 1.333)
        y_end = y_start + 45  # Updated from 34 for 1440p (34 * 1.333)
        for agent_place in range(0, 5):
            cropped_agent_image = screen_frame[y_start:y_end, 764:809]  # Updated from 573:607 for 1440p
            identified_agent = self.identify_agent(cropped_agent_image)
            all_agents.append(identified_agent)
            y_start = y_start + 45  # Updated from 34 for 1440p
            y_end = y_end + 45  # Updated from 34 for 1440p
        return all_agents

    def get_agents(self, main_frame):
        agents = {"top": [], "bottom": []}
        agents["top"] = self.process_frame(main_frame, "top")
        agents["bottom"] = self.process_frame(main_frame,"bottom")
        return agents


if __name__ == "__main__":
    get_score_board_agents = GetScoreBoardAgents()
    tab_images_directory = os.path.abspath(
        os.path.join(__file__, "../../test_images/Tab Images/"))
    for i in range(1, 7):
        start = time.time()
        image = cv2.imread('{}/{}.png'.format(tab_images_directory, i))
        print("=======================")
        print("File Number", i)
        agents = get_score_board_agents.get_agents(image)
        print("Agents:", agents)
        end = time.time()
        print("Time elapsed", end - start)
