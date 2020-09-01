# awesome-cost-learning-in-driving

This is the code to produce the results in the paper: "Analyzing the Suitability of Cost Functions for Explaining and Imitating Human Driving Behavior based on Inverse Reinforcement Learning" in ICRA 2020 by Maximilian Naumann, Liting Sun, Wei Zhan, Masayoshi Tomizuka.

The python files in the "tasks" folder contains the following three files:
1. SR.py
2. GL.py
3. EP.py

These three files are the main functions to generate the results in the paper. We have provided all feature calculation and learning via continuous-domain inverse reinforcement learning as in http://graphics.stanford.edu/projects/cioc/cioc.pdf.

Before you run the code, please first download the data from this link: https://drive.google.com/file/d/1SG_28MwfA98PTD-0-DVfLbdhtuHAjVtR/view?usp=sharing. The data is selected from the INTERACTION dataset. If you use the data in your future studies, please follow their instructions and rules on their website at http://interaction-dataset.com.