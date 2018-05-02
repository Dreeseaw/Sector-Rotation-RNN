'''
Sector Rotation Model Data Visualizer
William Dreese 2018

In Order
Healthcare: XLV - blue
Energy: XLE - red
Financials: XLF - green
Utilities: XLU - cyan
Tech: XLK - magenta
Consumer Disc: XLY - yellow
Consumer Staples: XLP - k
Materials: XLB - orange
Industrials: XLI - pink
Real Estate: IYR - brown
'''
import matplotlib.pyplot as plt
from SectorRotationDataPrep import DataPrep

class DataVis:
    def __init__(self):
        data = DataPrep()
        base = [a for a in range(0,data._weeks-1)]
        graph = [data._baseData[b] for b in range(len(data._baseData))]

        plt.plot(base,graph[0],"b-",
                 base,graph[1],"r-",
                 base,graph[2],"g-",
                 base,graph[3],"c-",
                 base,graph[4],"m-",
                 base,graph[5],"y-",
                 base,graph[6],"k-",
                 base,graph[7],"tab:orange",
                 base,graph[8],"tab:pink",
                 base,graph[9],"tab:brown")
        plt.ylabel('% change')
        plt.xlabel('weeks from April \'16')
        plt.axis([0, 100, -0.3, 0.7])
        plt.show()

dv = DataVis()
