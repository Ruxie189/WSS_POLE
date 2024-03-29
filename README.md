# WSS_POLE

This is a sample visualization from our PrOmpt cLass lEarning (POLE). Full training and inference codes will be released upon paper acceptance.

![](assets/WACV-24-1.png)

## Setup


1. Clone Repository:
```
git clone https://github.com/Ruxie189/WSS_POLE.git
```

2. Create a new python virtual environment as: 
```
python -m venv path/to/env/
```

3. Activate environment:
```
source path/to/env/bin/activate
```

4. Install dependencies:
```
pip install -r reqs.txt
```

## Running the Notebook

Open a terminal in the same directory as that of the cloned repository and run:
```
jupyter-notebook
```
Open ```sim_test.ipynb``` to find the codes for visualizing CLIP similarity. Running the notebook is mostly self explanatory and further instructions have been provided inside the notebook.


## Citations
```
@inproceedings{murugesan2024prompting,
  title={Prompting classes: Exploring the Power of Prompt Class Learning in Weakly Supervised Semantic Segmentation},
  author={Murugesan, Balamurali and Hussain, Rukhshanda and Bhattacharya, Rajarshi and Ben Ayed, Ismail and Dolz, Jose},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={291--302},
  year={2024}
}
```
