ALL_AGENTS = ["yoru", "astra", "killjoy", "omen", "raze", "reyna", "sage", "skye",
              "cypher", "jett", "breach", "brimstone", "kayo", "viper", "sova", "phoenix",
              "chamber", "fade", "neon", "gekko", "deadlock", "clove", "harbor", "iso"]
# TEMPLATE_CROP_COORDINATES = {"y_start":0,"y_end":33,"x_start":0,"x_end":100}
# FRAME_CROP_COORDINATES = {"y_start":107,"y_end":144,"x_start":1235,"x_end":1898}
# KILL_FEED_JUMP_HEIGHT = 43
# FEEDS_TO_EXAMINE = 6
# SIDES_TEMPLATES = ["blue_red","red_blue"]
MATCH_DETAILS_TEMPLATE = {"red": {}, "blue": {}, "events": []}
SAMPLE_COREGAME_DETAILS = {
    "score": [0, 0],
    "blue": [
        {
            "agent": "Raze",
            "name": "TheReal021#021",
            "rankName": "Bronze 1",
            "rr": 46,
            "leaderboard": 0,
            "level": "143",
            "team": "blue"
        },
        {
            "agent": "Skye",
            "name": "M0rty#7646",
            "rankName": "Silver 3",
            "rr": 5,
            "leaderboard": 0,
            "level": "128",
            "team": "blue"
        },
        {
            "agent": "Jett",
            "name": "im sad#yacs",
            "rankName": "Bronze 3",
            "rr": 91,
            "leaderboard": 0,
            "level": "76",
            "team": "blue"
        },
        {
            "agent": "Viper",
            "name": "Boosted Player#7126",
            "rankName": "Unrated",
            "rr": 0,
            "leaderboard": 0,
            "level": "28",
            "team": "blue"
        },
        {
            "agent": "Phoenix",
            "name": "NorthernLights#4907",
            "rankName": "Silver 2",
            "rr": 41,
            "leaderboard": 0,
            "level": "17",
            "team": "blue"
        }
    ],
    "red": [
        {
            "agent": "Raze",
            "name": "NeOx#7756",
            "rankName": "Unrated",
            "rr": 0,
            "leaderboard": 0,
            "level": "210",
            "team": "red"
        },
        {
            "agent": "Reyna",
            "name": "ASTA#LUFFY",
            "rankName": "Unrated",
            "rr": 0,
            "leaderboard": 0,
            "level": "156",
            "team": "red"
        },
        {
            "agent": "Viper",
            "name": "ARbuster23#5750",
            "rankName": "Bronze 2",
            "rr": 12,
            "leaderboard": 0,
            "level": "111",
            "team": "red"
        },
        {
            "agent": "Sage",
            "name": "DON#3664",
            "rankName": "Bronze 1",
            "rr": 81,
            "leaderboard": 0,
            "level": "41",
            "team": "red"
        },
        {
            "agent": "Jett",
            "name": "ItsPrithish#1234",
            "rankName": "Bronze 1",
            "rr": 66,
            "leaderboard": 0,
            "level": "32",
            "team": "red"
        }
    ]
}

corematch_example = {
    "blue": {
        "breach": {
            "agent": "breach",
            "leaderboard": 0,
            "level": "50",
            "name": "ETH3N#2688",
            "rankName": "Unrated",
            "rr": 0,
            "team": "blue",
            "alive": True,
            "health": 100, "weapon": "phantom"
        },
        "jett": {
            "agent": "jett",
            "leaderboard": 0,
            "level": "84",
            "name": "Ronny#2040",
            "rankName": "Silver 2",
            "rr": 10,
            "team": "blue",
            "alive": True,
            "health": 100, "weapon": "bucky"
        },
        "phoenix": {
            "agent": "phoenix",
            "leaderboard": 0,
            "level": "17",
            "name": "NorthernLights#4907",
            "rankName": "Unrated",
            "rr": 0,
            "team": "blue",
            "alive": True,
            "health": 100, "weapon": "ares"
        },
        "reyna": {
            "agent": "reyna",
            "leaderboard": 0,
            "level": "31",
            "name": "Meeseeks#7225",
            "rankName": "Bronze 1",
            "rr": 72,
            "team": "blue",
            "alive": True,
            "health": 100, "weapon": "shorty"
        },
        "viper": {
            "agent": "viper",
            "leaderboard": 0,
            "level": "61",
            "name": "Haniish5#3798",
            "rankName": "Unrated",
            "rr": 0,
            "team": "blue",
            "alive": True,
            "health": 100, "weapon": "marshall"
        }
    },
    "red": {
        "breach": {
            "agent": "breach",
            "leaderboard": 0,
            "level": "57",
            "name": "Modi#3216",
            "rankName": "Bronze 1",
            "rr": 60,
            "team": "red",
            "alive": True,
            "health": 100, "weapon": "judge"
        },
        "killjoy": {
            "agent": "killjoy",
            "leaderboard": 0,
            "level": "66",
            "name": "Reflex#2886",
            "rankName": "Unrated",
            "rr": 0,
            "team": "red",
            "alive": True,
            "health": 100, "weapon": "ghost"
        },
        "phoenix": {
            "agent": "phoenix",
            "leaderboard": 0,
            "level": "64",
            "name": "Jose#7490",
            "rankName": "Silver 3",
            "rr": 10,
            "team": "red",
            "alive": True,
            "health": 100, "weapon": "spectre"
        },
        "raze": {
            "agent": "raze",
            "leaderboard": 0,
            "level": "100",
            "name": "Riyan Kc#6969",
            "rankName": "Bronze 1",
            "rr": 78,
            "team": "red",
            "alive": True,
            "health": 100, "weapon": "stinger"
        },
        "reyna": {
            "agent": "reyna",
            "leaderboard": 0,
            "level": "130",
            "name": "ChaModDunga#6480",
            "rankName": "Silver 2",
            "rr": 98,
            "team": "red",
            "alive": True,
            "health": 100, "weapon": "vandal"
        }
    }
}

agents_ultimate_points = {"phoenix": 6, "jett": 8, "viper": 9,
                          "sova": 8, "cypher": 6, "omen": 7,
                          "raze": 8, "breach": 9, "sage": 8,
                          "brimstone": 8, "reyna": 6, "killjoy": 9,
                          "astra": 7, "yoru": 7, "kayo": 8,
                          "skye": 8,"chamber": 8,"fade": 8,
                          "neon": 7,"gekko": 7,"deadlock": 7,
                          "clove": 7,"harbor": 7,"iso": 7}
