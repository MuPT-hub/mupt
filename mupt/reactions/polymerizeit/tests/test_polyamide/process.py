import argparse

import polymerizeit as pi
from mupt.mutation.polymerizeit.gmx_simple import GMXsimple
from mupt.mutation.polymerizeit.gmx_simple import create_parser


# Standard boilerplate to call the main() function to begin the program.
if __name__=="__main__":
    try:
		
        args=create_parser().parse_args()
        param_args=vars(args)
        process=GMXsimple(**param_args)
        process.preprocess()
    except (RuntimeError, TypeError, NameError) as e:
       print(e)
       print("Something went wrong in checking the reactions")	
