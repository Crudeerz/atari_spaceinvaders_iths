# Spaceinvaders Laboration at ITHS
This repo is for laboration at ITHS, training an agent using Deep-Q Reinforcement Learning, to play the game Spaceinvaders.

[Assignment Paper](https://github.com/Crudeerz/atari_spaceinvaders_iths/blob/main/Laboration/Assignment/Laboration_Djupinl%C3%A4rning_HT24.pdf)

## View project presentation
The laboration is presented in a small app using streamlit.  
The app is deployed to stremlit cloud and can be viewed here  

[Streamlit App](https://atarispaceinvadersiths-gbtruuur5cipwykf6mhw7z.streamlit.app/)  

To run the app locally: 
- Clone this repo
- Install streamlit using pip
- Run the app from the directory containing "app.py" 

```bash
$ pip install streamlit
$ cd <root directory of project>
$ streamlit run app.py
```

## Download the repo and play with the code

- Clone this repo 
- Use pip to install requirements

 ```bash 
 $ pip install -r requriements.txt
 ```

### Run the code
The python file **"Laboration/train_agent.py"** is the main file that starts the script and trains the agent.   
When running the script, it starts to train the model and regularly saves modelfiles to a locally created directory: **/Local/Models**  
Default, it also saves plots using matplotlib every 10th episode to: **/Local/Plots**, this can be changed using the **"SAVE_PLOTS"** variable.


> [!NOTE]
> The saving of plots is only made for the sake of the streamlit app.
> If you would like to construct a ".gif" from all saved plots in */Local/Plots*, run the program **"Laboration/Resources/make_gif.py"**  
> By default, this program removes the images from the source directory when the gif has been successfully created.

If you want to test the models that has been saved and record them playing the game, you can run **"Laboration/record_agent.py"**.  
Be sure to change the path to which model you want to record.
