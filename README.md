# GrabCut-Annotation-Tool
Annotation tool that leverages OpenCV's GrabCut implementation.  
It can be used to create binary masks for salient objects.  
Due to GrabCut's algorithm, it is suitable for annotation of data with clear boundaries.<br>

# User Guide
## Prerequisites
* [Python](https://www.python.org/downloads/) (3.8 or later) (suggested installation via [Pyenv](https://github.com/pyenv/pyenv#installation))
* [Poetry](https://python-poetry.org/docs/#installation) (1.1.12 or later)
## Installation
Run ```poetry install``` from within project's root.

## Usage
Run ```python main.py``` from within project's root.  
For a list of possible options run ```python main.py --help``` 

## Directories
Following are the default directories:
* ```GrabCut-Annotation-Tool/input``` Input Images
* ```GrabCut-Annotation-Tool/output/image``` Output Images
* ```GrabCut-Annotation-Tool/output/annotation``` Output Masks  

These can be customized when the app is run.  

## Output naming convention
Given an input image ```<filename>.<filetype>```: 
* the output image will be named ```<filename>_<class>.png``` 
* the output mask will be named ```<filename>_<class>_mask.png```

# Using GrabCut-Annotation-Tool
### File select
You can switch the annotation target by clicking the file list.<br>
keyboard shortcut 　↑、p：preview file　↓、n：next file<br>
<img src="https://user-images.githubusercontent.com/37477845/131686101-c94132bc-4b76-488a-85fe-69d9d9c216bd.png" width="80%">

### Initial ROI designation
You can specify the initial ROI by right-drag the mouse when "Select ROI" is displayed.<br>
<img src="https://user-images.githubusercontent.com/37477845/131687291-4f4c06d5-89fa-452d-925f-5576edc5af64.png" width="80%"><br><br>

After the drag is finished, GrabCut processing is performed.<br>
<img src="https://user-images.githubusercontent.com/37477845/131687690-295dc463-f82e-447b-86f8-65bbf6cf4e2d.png" width="80%"><br><br>

The area is selected.<br>
<img src="https://user-images.githubusercontent.com/37477845/131688127-3fc1c00e-0f99-435a-aa29-d9392c7af6d0.png" width="80%"><br><br>

### Background designation
You can specify the background by dragging the right mouse button.<br>
<img src="https://user-images.githubusercontent.com/37477845/131688309-c47184d9-f793-49f0-aa26-445ea2c2b431.png" width="80%"><br><br>

<img src="https://user-images.githubusercontent.com/37477845/131688599-dc78e307-8a3b-4ec7-a9be-05325486ee5e.png" width="80%"><br><br>

### Foreground designation
You can switch to foreground specification by unchecking "Mark background".<br>
keyboard shortcut　Ctrl<br>
<img src="https://user-images.githubusercontent.com/37477845/131688947-ab0505ca-8413-4afe-8d5a-c42ae1f25a3f.png" width="80%"><br><br>

You can specify the foreground by dragging the right mouse button.<br>
<img src="https://user-images.githubusercontent.com/37477845/131689310-5447308d-2019-48d7-8a43-df7707969599.png" width="80%"><br><br>

<img src="https://user-images.githubusercontent.com/37477845/131689509-ea0597a4-939a-4821-a077-40720687e8b1.png" width="80%"><br><br>

### Saving
To save the image and the mask click the **Save** button, a prompt will ask you to type in the class id for the masked object. Upon confirmation the results are save.

# Credits
Kazuhito Takahashi(https://twitter.com/KzhtTkhs) for the original implementation from which this project is derived.
 
# License 
GrabCut-Annotation-Tool is under [Apache-2.0 License](LICENSE).

The sample image uses the photograph of [フリー素材 ぱくたそ](https://www.pakutaso.com).
