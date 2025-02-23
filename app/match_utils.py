import constants
from components.get_corematch import GetCoreMatch


class MatchUtils():
    # def __init__(self,match_details):
    #     self.match_details = match_details

    def instantiate_match_details(self):
        match_details = GetCoreMatch().fetch_match_details()
        match_details = self.set_ultimate_points(match_details)
        if match_details["initiated_player_team"] != "blue":
            match_details = self.switch_sides(match_details)
        match_details['score'] = [0, 0]
        return match_details

    def set_ultimate_points(self, match_details):
        for side in ["blue", "red"]:
            for key, value in match_details[side].items():
                value["required_ultimate_points"] = constants.agents_ultimate_points[value["agent"]]
        return match_details

    def update_match_details(self, match_details, new_data):
        for side in ["blue", "red"]:
            for index, (key, value) in enumerate(match_details[side].items()):
                if value["agent"] in [agent.lower() for agent in new_data["alive_agents"][side] if agent is not None]:
                    value["alive"] = True
                    if side == "blue":
                        value["health"] = new_data["health_values"][side][index]
                    else:
                        value["health"] = 100
                else:
                    value["alive"] = False
                    value["health"] = 0
                if "agents_with_loadouts_shields" in new_data:
                    for agent_information in new_data["agents_with_loadouts_shields"][side]:
                        print("agent_information", agent_information, side)
                        if agent_information[0].lower() == value["agent"].lower():
                            value["weapon"] = agent_information[1]
                            value["shield"] = agent_information[2]
                            ultimate_info = agent_information[3]
                            if isinstance(ultimate_info, dict):
                                value["current_ultimate_points"] = ultimate_info.get("number", None)
                            elif isinstance(ultimate_info, str):
                                # When ultimate info is a string (e.g., "READY"), store it as is.
                                value["current_ultimate_points"] = ultimate_info
                            else:
                                value["current_ultimate_points"] = ultimate_info

        return match_details

    def switch_sides(self, match_details):
        # Change blue to red and vice versa.
        old_red_values = match_details["red"]
        old_blue_values = match_details["blue"]
        match_details.pop("red")
        match_details.pop("blue")
        match_details["red"] = old_blue_values
        match_details["blue"] = old_red_values
        for side in ["blue", "red"]:
            for key, value in match_details[side].items():
                value["side"] = side

        return match_details

    def end_match(self):
        # post request to live feed api to stop video reception
        # response = requests.post('http://localhost:4444/stop_stream_reception', json={"key": "value"})
        # print( response.status_code)
        # return constants.MATCH_DETAILS_TEMPLATE
        # stop match and reset match details to template
        pass
