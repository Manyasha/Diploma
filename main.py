import argparse
from testFunctions import f, x0, x_star, f_star
from secondaryFunctions import printInfo
from conjugateGradientMethods import fourStepsCGM, threeStepsCGM

parser = argparse.ArgumentParser(description='Modifications of Conjugate Gradient Method')

parser.add_argument('-n', nargs='+', type=int, help='List of numbers of test functions which will be processed')
parser.add_argument('-x0', nargs=1, help='Custom init point x0')
parser.add_argument('-eps', nargs='?', default='0.01', help='Accuracy of calculations')
parser.add_argument('-a', action='store_true', help='Process all test functions')

args = parser.parse_args()

def main():
    if (not args.n and not args.a):
        print('Please, use -n or -a argument. For details see --help')
        return
    elif (args.n and args.a):
        print('Only one argument from -n, -a can be used')

    for i in range(1, len(f) + 1) if args.a else args.n :
        if i < 1 or i > len(f) + 1:
            print('Invalid function index i =', i)
            continue
        x = args.x0 if args.x0 else x0[i]
        
        fourStepsRes = fourStepsCGM(f[i], x0, args.eps)
        threeStepsRes = threeStepsCGM(f[i], x0, args.eps)

        printInfo(f[i], x0, {'x_star': x_star[i], 'f_star': f_star[i]}, fourStepsRes, threeStepsRes)

main()


    
