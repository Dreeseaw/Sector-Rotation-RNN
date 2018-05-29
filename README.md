# Sector Rotation Model
### by William Dreese

A simple recurrent neural network that uses weekly changes in 10 major sector ETFs to predict which sectors will grow in the coming weeks.

The model is currently a 2-layer IndRNN trained with basic SGD, based off the paper 'Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN'. The arVix link is: https://arxiv.org/abs/1803.04831.

The theory behind the model is that if there are relations between the growth patterns of multiple sectors, and by recognizing these relations, then I can make accurate predictions as to the next week(s) for all 10 sectors. 

Using data from alphavantage.com, 10 sector ETFs (Tech, Finacial, etc) are analyzed by feeding in the percent change of the past week and training it to predict the next week's percent change.

This model could be applied to the portfolio management strategy of sector rotation, where investors target specific sectors for growth upside compared to other areas of the market. Another application of this model is to target a sector for a short-term swing trade, using the recurrency of the network to generate exact movement predictions for the next X weeks. 

The sectors and their respective ETFs are
Healthcare: XLV,
Energy: XLE,
Financials: XLF,
Utilities: XLU,
Tech: XLK,
Consumer Disc: XLY,
Consumer Staples: XLP,
Materials: XLB,
Industrials: XLI,
Real Estate: IYR

All .txt files are used to store data, as alphavantage has a (annoyingly-low) limit on api call requests. 

The Data Prep class is used to parse out the csv files from alphavantage (when you run it for current data), as well as prepare it to train the model. 

Sector Rotation Visuals utilizes matplotlib to graphically display the sector returns for the past two years. 

The model lies in the SectorRotationModel file, which uses the class from the Data Prep file for it's training and testing data.  
