import os
import Calibration


def main():
    option_file_path = 'C:\\Users\\pawon\\Dropbox\\Option_pricing_USO\\Data\\Processed'
    option_file = os.path.join(option_file_path, 'USO_European_converted_20221110_IV.csv')
    rn_params = [0.08759123209373104,-0.04558330770874242,-0.05113411797116935, -0.05944401576211998, 0.03246903959485084]
    test = Calibration.Calibration(option_file, ('SVCJ', 'NL', 0.5), rn_params=rn_params)
    test.option_test()
    #test.calibrate()

if __name__ == "__main__":
    main()



