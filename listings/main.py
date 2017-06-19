import argparse
from numpy import set_printoptions
from testFunctions import f, x0, x_star, f_star
from secondaryFunctions import printInfo, showPlot
from conjugateGradientMethods import fourStepsCGM, threeStepsCGM, 
							  nonQvadFourStepsCGM, nonQvadThreeStepsCGM

set_printoptions(precision=5, suppress=True)

parser = argparse.ArgumentParser(
		 description='Modifications of Conjugate Gradient Method')

parser.add_argument('-n', nargs='+', type=int, 
		help='List of numbers of test functions which will be processed')
parser.add_argument('-x0', nargs='+', type=float, 
		help='Custom init point x0')
parser.add_argument('-eps', nargs='?', type=float, default='0.000001', 
		help='Accuracy of calculations')
parser.add_argument('-a', action='store_true', 
		help='Process all test functions')
parser.add_argument('-t', action='store_true', 
		help='Print test data')
parser.add_argument('-p', action='store_true', help='Show plot')


args = parser.parse_args()

def main():
    if (args.t):
        for i in range(1, len(f) + 1):
            message = "Function number: " + str(i)
            print(message)
            printInfo(f[i], x0[i], args.eps, 
					{'x_star':x_star[i],'f_star':f_star[i]},{},{})
        return    
    
    if (not args.n and not args.a):
        print('Please, use -n or -a argument. For details see --help')
        return
    elif (args.n and args.a):
        print('Only one argument from -n, -a can be used')

    for i in range(1, len(f) + 1) if args.a else args.n :
        if i < 1 or i > len(f) + 1:
            print('Invalid function index i =', i)
            continue
        x_start = args.x0 if args.x0 else x0[i]
        x_start = x_start if type(x_start) == list and 
				  type(x_start[1]) == list else [x_start]

        for x_start_j in x_start:
            fourStepsRes = fourStepsCGM(f[i], x_start_j, args.eps)
            threeStepsRes = threeStepsCGM(f[i], x_start_j, args.eps)

            nonQvadFourStepsRes=nonQvadFourStepsCGM(f[i],x_start_j,args.eps)
            nonQvadThreeStepsRes=nonQvadThreeStepsCGM(f[i],x_start_j,args.eps)

            printInfo(f[i], x_start_j, args.eps, 
		{'x_star': x_star[i], 'f_star': f_star[i]}, fourStepsRes, threeStepsRes)
            printInfo(f[i], x_start_j, args.eps, 
		{'x_star':x_star[i],'f_star':f_star[i]},nonQvadFourStepsRes,nonQvadThreeStepsRes)
            if args.p:
                showPlot(f[i], x_start_j, args.eps, 
							fourStepsRes, threeStepsRes)
                showPlot(f[i], x_start_j, args.eps,
							nonQvadFourStepsRes, nonQvadThreeStepsRes)

main()


    
