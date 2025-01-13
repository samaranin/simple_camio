Simple CamIO 2D

Description: Simple CamIO 2D is a Python version of CamIO specialized to a flat, rectangular model such as a tactile map. This version relies on finger/hand tracking rather than the use of a stylus.

Requirements: To run Simple CamIO 2D, one needs to set up several things. 
- Firstly, There needs to be a json file that defines a model, that is it describes the components of an interactive map.  It contains the filenames of the various components of a model, as well as other important information such as the hotspot dictionary.  An example we recommend using as reference is models\UkraineMap\UkraineMap.json.

- Secondly, we require a printed map with text features along all four edges. An image of this map should be included, with its filename being specified in the element "template_image" of the model dictionary of the input json file.  We recommend using models\UkraineMap\template.png as an example to print out.

- Next, we require a digital version of the map that represents hotspot zones with unique indices as in models\UkraineMap\UkraineMap.png, and this filename should be specified in the element "filename" of the model dictionary of the input json file. Each index is a specific (R,G,B) color value. The image dimensions should match that of the template image. 

- Sound files, as named in the hotspots dictionary in the supplied json file, should be placed in the appropriate folder, as specified in the hotspots dictionary. The hotspots dictionary maps the zone index (from the zone map) to the sound file.

- Python 3.9 installed with opencv, numpy, scipy, mediapipe, and pyglet libraries (most of which can be installed through Anaconda, except mediapipe and pyglet which need to be installed via pip). The required library versions are specified in the requirements.txt file.

For best performance, we recommend the camera sit above the map to get a fronto-parallel view as much as possible. The camera should have an unobstructed view of the 4 sides of the map, and the hand should be held such that the camera can clearly view it. The map should be close enough to the camera to take up most of the space in the camera image (so it is well resolved), but sufficient space (roughly 20 cm) between the map and the edges of the image should be available to ensure reliable finger/hand tracking even when the user is pointing to a feature near an edge of the map.

To run, simply run the simple_camio.py script as "python simple_camio.py --input1 \<json file\>" where \<json file\> is the location of the json file containing the model parameters, such as models/UkraineMap/UkraineMap.json. 

To reset the homography, that is, to update the map position in the image, press the 'h' key.

To use, simply make a pointing gesture by extending the index finger out and curling in the other fingers.  The area on the map indicated under the tip of the index finger will be dictated aloud.  The hand should be kept flat against the surface with the finger jutting out rather than the hand being held up above the map with the finger pointed down.
![](img/pointing_yes.jpg) ![](img/pointing_no.jpg)

__________________________________________________
How to install Python libraries
1. Download and install Python 3.9.13 from https://www.python.org.
2. From the command line, in your working directory, type and enter "python -m venv camio"
3. Then type and enter "camio\Scripts\activate"
4. Then type and enter "git clone https://github.com/Coughlan-Lab/simple_camio.git"
5. Then type and enter "cd simple_camio" followed by "git fetch" and "git checkout simple_camio_2d"
6. Then type and enter "pip install -r requirements.txt"
