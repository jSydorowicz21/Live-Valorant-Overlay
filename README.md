
# Valorant Live Overlay

An Overlay proof-of-concept Application for Valorant giving information in real-time about current match, spike and player status(dead/alive, weapons, ultimate points, shields etc.)

### Important Caveat:

- Currently works only with the following configuration :

- "Fullscreen Windowed"
-  [1920x1080](https://github.com/tac-eillA/Valorant-Match-Overlay/tree/main)
    **REMOVED 1920x1200 support due to lack of use**
- "Fill" Mode
- **Multi monitor support does not appear to function at the moment and a hybrid igpu+dgpu setup will cause issues with CUDA processing. App will still work with hybrid gpu but will process using cpu rather than gpu**

## Disclaimer

This is a fan-made project. Riot Games does not endorse, sponsor or affiliate with this project. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

## Description

This project is a proof-of-concept that gets you live details of a match and populates a HTML file which can be used as an overlay in streaming apps such as OBS, and vMix
  
## Installation

#### Prerequisites

- Python3.x **MUST BE 64 BIT PYTHON**

- Run the following command in the project's root directory (assuming you have cloned this repository) to install all other dependencies

```sh
pip install -r requirements.txt
```

## Usage

1. Run Python Server

Change the working directory to ```app``` and run the following command to start the python server which will serve requests such as starting/stopping match processing and facilitate communication using Socket IO between different components (both the backend and the frontend).

```
python app.py
```

2. Open Overlay

##### Using the Overlay in streaming apps

For using the overlay with OBS, add a "BrowserSource" and navigate to the ```overlay``` directly and open ```index.html``` . Set the width and height to 1920 and 1080. It should show dummy data for two teams. If you are using vMix, add a "Browser" input and type the hard file path to ```index.html``` in the url window. Set your height and width just as with OBS and you will receive the same dummy data.

3. Instruct server to start match processing

When your match has just started, send a GET/POST request to ```http://127.0.0.1:4445/start_match``` or just open this link in the browser. It will start processing the match and the details will be reflected in the overlay.

4. Stop Match Processing

When your match has ended, send a GET/POST request to ```http://127.0.0.1:4445/stop_match``` or just open this link in the browser. It will stop processing the match.

#### Get Match Details from the Server directly

You can make POST requests to ```http://127.0.0.1:4445/get_match_details``` to get the current match details.

To receive match details the moment they are updated, connect to Socket IO at ```http://127.0.0.1:4445/``` (see app/components/get_live_frames.py to see how to connect) and listen on "new_event" event.

## How does it work?

### Match Processing Techniques

This app uses Image Processing using OpenCV and largely Template Matching to get information from a screenshot of the game.

#### Starting the match processing

- When you send the "start_match" request to the server, it fetches all the players' information such as their name, rank, team etc.

- It then starts another process of taking screenshots (see app/components/get_live_frames.py) of the game every 1 second and passes this screenshot to the relevant module for processing.

#### What happens in the match processing

- The agent icons' present in the header bar are matched with all of the existing player icon templates (see app/components/header_agents.py) and the one that matches the most (a confidence of greater than 70%) is then determined to be alive.

- All of the other data is grabbed from the scoreboard (when it is present) using a combination of Template Matching and OCR. You can inspect all the respective modules to see how they work.

### Populating the Overlay

The Overlay files (overlay/index.html+overlay/app.js) connect to the Python server via SocketIO.

The provided overlay files are a crude proof of concept of a working overlay.

#### Developing your own overlay

Feel free to develop your own overlay using the data received from the server.

## Possible Future Updates

- CREDS through OCR

- Score through in-game API

- UI for application:

- Inputting team's names, logos etc.

- Automatic match processing start/end using presence data

- Getting information from all modules parallelly to speed up information retrieval.

- Write resizing logic to support different resolutions.

## Credits

The players' information retrieval method is inspired by the wonderful work done by [isaacKenyon's VRY](https://github.com/isaacKenyon/VALORANT-rank-yoinker), [colinhartigan's fork](https://github.com/colinhartigan/valorant-live-match-rank-grabber), and the groundwork done by [deepsidh9](https://github.com/deepsidh9).
