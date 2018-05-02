'''
Sector Rotation Data Prep
William Dreese 2018

manual data prep
- get csv
- use excel to delete all columns
-297 should be first blank col
- open file with a text editor, copy and paste it into .txt file
- ETF_NAMEdata.txt (ex: "XLVdata.txt")

string for data:
https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=XLV&outputsize=full&apikey=RH214J1I303UHRD9&datatype=csv

Take AlphaVantage data on 10 SPDR sector ETFs
Healthcare: XLV -
Energy: XLE -
Financials: XLF -
Utilities: XLU -
Tech: XLK -
Consumer Disc: XLY -
Consumer Staples: XLP -
Materials: XLB -
Industrials: XLI -
Real Estate: IYR

-get %change for each trading weeks close to close price from last 2 years (100)
-compile all 10 datasets into 

'''
ETFs = ["XLV","XLE","XLF","XLU","XLK","XLY","XLP","XLB","XLI","IYR"]

class DataPrep:
    def __init__(self):
        
        self._weeks = 100
        self._etfs = 10
        
        self._data      = [self.dataTrim(self.parseFile(ETFs[a])) for a in range(self._etfs)]
        self._baseData  = [self.baseTrans(a) for a in range(self._etfs)]
        self._deltaData = [self.deltaTrans(a) for a in range(self._etfs)]
        
        deltaBinaryData = [self.deltaBinaryTrans(a) for a in range(self._etfs)]
        newDelta = self.list2dRotate(self._deltaData)
        newNorm = [self.normalize1dList(newDelta[x]) for x in range(len(newDelta))]
        newBin = self.list2dRotate(deltaBinaryData)
        newBin = newBin[1:]
        totalData = list(zip(newNorm,newBin))
        
        self._trainingData = totalData[:int(float(len(totalData))*2.0/3.0)]
        self._testData = totalData[int(float(len(totalData))*2.0/3.0):]

    def normalize1dList(self,data):
        return [(data[a]-min(data))/(max(data)-min(data)) for a in range(len(data))]

    def list2dRotate(self,list2d):
        ret = []
        for x in range(0,len(list2d[0])):
            add = []
            for y in range(0,len(list2d)):
                add += [list2d[y][x]]
            ret += [add]
        return ret

    def dataTrim(self,lis):
        while len(lis) > self._weeks:
            lis.pop()
        return lis

    def deltaBinaryTrans(self,etf):
        lis = self._deltaData[etf]
        adder = []
        for a in range(0,len(lis)):
            if lis[a] > 0.0: adder += [1]
            else: adder += [0]
        return adder

    def baseTrans(self,etf):
        lis = self._data[etf]
        baseData = [lis[-b] for b in range(1,len(lis))]
        base = baseData[0]
        for a in range(0,self._weeks-1):
            baseData[a] = (baseData[a] - base) / base
        return baseData

    def deltaTrans(self,etf):
        lis = self._data[etf]
        baseData = [lis[-b] for b in range(1,len(lis))]
        ret = []
        for a in range(1,self._weeks-1):
            base = baseData[a-1]
            ret += [(baseData[a] - base) / base]
        return ret

    def parseFile(self,etf):
        etfStr = etf + "data.txt"
        r = open(etfStr)
        s = r.readlines()
        
        adder = []
        for a in range(1,len(s)):
            f = s[a]
            if f.find(",") != -1: f = f[4:]
            datway = f.rstrip()
            adder += [float(datway)]
        return adder

dp = DataPrep()
