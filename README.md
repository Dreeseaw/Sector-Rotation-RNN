# Sector Rotation RNN
### by William Dreese

A simple recurrent neural network that uses weekly changes in 10 major sector ETFs to predict which sectors will grow in the coming weeks.

Using data from alphavantage.com, 10 sector ETFs (Tech, Finacial, etc) are analyzed by first normalizing the weekly change between the 10 closing prices from 0.0 to 1.0. This normalized vector is paired with a vector of 10 1's or 0's, representing if the sector grows (1) or shrinks (0) in the coming week. These values are then used to train a 3-layer (in-hidden-out) recurrent model, featuring a 10-node hidden layer. 

This model could be applied to the portfolio management strategy of sector rotation, where investors target specific sectors for growth upside compared to other areas of the market. Another application of this model is to target a sector for a short-term swing trade, using the recurrency of the network to generate exact movement predictions for the next X weeks. 

The sectors and their respective ETFs are
Healthcare: XLV
Energy: XLE
Financials: XLF
Utilities: XLU
Tech: XLK
Consumer Disc: XLY
Consumer Staples: XLP
Materials: XLB
Industrials: XLI
Real Estate: IYR
