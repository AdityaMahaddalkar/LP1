import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time
from functools import reduce
import warnings
import copy
warnings.filterwarnings("ignore")


'''

GLOBALS

'''
def convert_date(date_bytes):
		return mdates.strpdate2num('%Y%m%d%H%M%S')(date_bytes.decode('ascii'))

def percentChange(startPoint, currentPoint):
	try:
		x = ((currentPoint-startPoint)/abs(startPoint))*100
		if x == 0.0:
			return 0.00000000001
		else:
			return x
	except:
		return 0.0000000001


date, bid, ask = np.loadtxt('resources/GBPUSD1d.txt', unpack=True,
		delimiter=',', converters={0:convert_date})

patternAr = []
performanceAr = []
avgLine = (bid+ask)/2
patForRec = []

def patternStorage():

	global patternAr, performanceAr, avgLine, patForRec

	patStartTime = time.time()

	x = len(avgLine) - 60

	y = 31
	while y < x:

		p = []
		for _ in range(30):
			p.append(percentChange(avgLine[y-30], avgLine[y-(29 - _)]))

		outcomeRange = avgLine[y+20:y+30]
		currentPoint = avgLine[y]

		try:
			avgOutcome = reduce(lambda x, y: x+y, outcomeRange)/len(outcomeRange)
		except Exception as e:
			print(e)
			avgOutcome = 0

		futureOutcome = percentChange(currentPoint, avgOutcome)
		
		patternAr += p
		performanceAr.append(futureOutcome)
		y+= 1

	patEndTime = time.time()
	print(f'Length of patternAr {len(patternAr)}')
	print(f'Length of performanceAr {len(performanceAr)}')
	print(f'Pattern storage took {patEndTime - patStartTime}')

def graphRawFX():

	global patternAr, performanceAr, avgLine, patForRec

	date, bid, ask = np.loadtxt('resources/GBPUSD1d.txt', unpack=True,
		delimiter=',', converters={0:convert_date})

	fig = plt.figure(figsize=(10, 7))
	ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)
	ax1.plot(date, bid)
	ax1.plot(date, ask)

	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))

	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	
	plt.subplots_adjust(bottom=.23)

	ax1_2 = ax1.twinx()
	ax1_2.fill_between(date, 0, (ask-bid), facecolor='g', alpha=.3)

	plt.grid(True)
	plt.xticks(rotation=75)
	plt.show()


def currentPattern():

	global patternAr, performanceAr, avgLine, patForRec

	cp = []
	for _ in range(30):
		cp.append(percentChange(avgLine[-31], avgLine[-(30-_)]))

	patForRec = copy.deepcopy(cp)

	print(patForRec)


def patternRecognition():

	global patternAr, performanceAr, avgLine, patForRec

	for eachPattern in patternAr:
		sim = []
		for _ in range(30):
			sim.append(100.0 - abs(percentChange(eachPattern[_], patForRec[_])))

		howSim = sum(sim)/30

		if howSim > 40:
			patdex = patternAr.index(eachPattern)
			print('###########################')
			print(patForRec)
			print('===========================')
			print(eachPattern)
			print('===========================')
			print(f'Predicted Outcome {performanceAr[patdex]}')
			xp = np.arange(1, 31, 1)
			fig = plt.figure()
			plt.plot(xp, patForRec)
			plt.plot(xp, eachPattern)
			plt.show()
			print('###########################')


'''

TEST DRIVER

'''

if __name__ == '__main__':
	# graphRawFX()
	totalStart = time.time()
	patternStorage()
	currentPattern()
	patternRecognition()
	print(f'Total execution time {time.time() - totalStart}')