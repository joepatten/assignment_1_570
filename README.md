# Assignment 1
Joe Patten

This assignment was done 100\% all by myself. 
Note: the figures and output might be a bit different than what is written in the pdf as I ran the code many times and did not feel the need to updated my pdf (as the figures were generally similar to previous versions).

## run_code.sh
In order to run the code for assignment 1, you can either run the run_code.sh shell script, or run `python ./Assignment_1.py` in your shell. 
Running the shell script in linux was a lot smoother than running it in windows (although both worked, for some reason running it on windows didn't print out some of my print statements, and thus did not let me know the progress of the code). 

## Assignment_1.py
Line 11: The data is downloaded, unzipped, and converted to numpy arrays. 
5.2 runs a bit slower (and is a part of code I wish to optimize in the future). It does run though (it just takes forever to do so). 
If you want to test out 5.2 code, then I would suggest changing the variables `iterations_a`, and `iterations_b_d` to be lower numbers. 


## Github repo
A link to the github repo can be found here: https://github.com/joepatten/assignment_1_570

Things to do:
- Optimize MC algorithms (use sparse matrices?)
- Add bias term

## Language
Python 3.6 is required

## Packages Used
- pandas 0.23.4
- numpy 1.16.1
- matplotlib 3.0.2
- tensorflow 1.13.1
- wget 3.2


## Output
The output (saved as csv files) for problem 5.1 is saved in the ./output/5_1/ folder, and the output for problem 5.2 is saved in the ./output/5_2/ folder.

## Figures
The figures (saved as png files) for problem 5.1 are saved in the ./figure/5_1/ folder, and the output for problem 5.2 are saved in the ./figure/5_2/ folder.
