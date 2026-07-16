import os
import subprocess as sp
import sys
import argparse
from datetime import datetime
import json

import polymerizeit as pi
from polymerizeit.utils.log_file_contents import write_log_file_header, write_log_file_body

class GMXsimple(pi.ProtocolBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)	
        self.__dict__.update(kwargs)
        self.params=kwargs
    
    def run(self,start,end,rewrite=False):
        #Checking if the preprocessed inputs are available
        #If not, then create the file
        if not os.path.exists("processed_inputs.json"):
            self.preprocess()
 
        else:     
            preprocessed_inputs_info=self.preprocessed_inputs_info 
        
        write_log_file_header(self.logfile) 

        for cycle in range(start,end):

            #Energy minimization and equilibrtion of structure for the current cycle
            #Uses MD engine files
            if not os.path.exists("./emin/emin-whole-iter%d.gro" % (cycle, )) or rewrite:
                sp.call(['bash ./emin.sh %d' % (cycle, )], shell = True)

            if not os.path.exists("./equil/equil-whole-iter%d.gro" % (cycle, )) or rewrite:
                sp.call(['bash ./equil.sh %d' % (cycle, )], shell = True)

            log=""
            rx_type=""

            
            #Updating the iteration-specific file names in pre-processed info
            preprocessed_inputs_info['ifgro']="./equil/equil-iter"+str(cycle)+".gro"
            preprocessed_inputs_info['ofgro']="./gro-files/iter"+str(cycle+1)+".gro"
            preprocessed_inputs_info['iftop']="./topology/iter"+str(cycle)+".top"
            preprocessed_inputs_info['oftop']="./topology/iter"+str(cycle+1)+".top"

            #Mandatory: Create sqlite tables from the updated MD engine files after equilibration
            #Uses the pre-processed info, which is passed in the constructor
            #In this protocol, input files are in gromacs format
            pi.GenSystemTables(preprocessed_inputs=preprocessed_inputs_info).from_gromacs()
            
            #Check for the reaction 
            chk_rxn=pi.CheckReactionStandard(preprocessed_inputs=preprocessed_inputs_info).find_eligible_reactions()
            if chk_rxn[0] == 1:
                print("New bonds formed in the normal routine. Now writing updated files ...")
                pi.WriteGromacsFiles(preprocessed_inputs=preprocessed_inputs_info).run()
                rx_type="normal routine"
                log=chk_rxn[1],chk_rxn[2]
                write_log_file_body(self.logfile,log,cycle+1,rx_type)
            else:
                #copy the file to next iteration if no reaction occurss
                sp.call(['bash ./copy_previous_iter_files.sh %s' % (cycle, )], shell = True)
                log=0,"None"
                rx_type="no reaction"
                write_log_file_body(self.logfile,log,cycle+1,rx_type)
                
        return None
    
    def preprocess(self):
        preprocessed_inputs_info=pi.PreprocessReactionInputs(**self.params,
                                                                out_itp_file_path="./topology",
                                                                out_gro_file_path="./gro-files").process_reaction_info()
        self.set_preprocessed_inputs_info(preprocessed_inputs_info=preprocessed_inputs_info)
   
def create_parser():
	parser = argparse.ArgumentParser(prog = 'PolymerizeIt! preprocessing', usage = '%(prog)s [-h for help]', \
                                             description='Generate input files with unique atom IDs')
	parser.add_argument('-preprocess_database', '--preprocess_database', help = 'Input self.preprocess_database file name file', default="init.db")
	parser.add_argument('-main_database', '--main_database', help = 'Input self.preprocess_database file name file', default="main.db")
	parser.add_argument('-preprocess_only', "--preprocess_only", help = "If true, only preprocess the inputs.", default = bool(False))
	parser.add_argument('-inputs_file',"--inputs_file", help="Parameters needed for polymerization")
	parser.add_argument('-iftop', help = 'Input .top/.itp file (Required).')
	parser.add_argument('-ifgro', help = 'Input .gro file (Required).')
	parser.add_argument('-oftop', "--oftop", help = 'Output .top/.itp file. \
                        Generated only when there is a reaction.', default = "rxn.top")
	parser.add_argument('-ofgro', "--ofgro", help = 'Output .gro file. \
                        Generated only when there is a reaction.', default = "rxn.gro")
	parser.add_argument('-nangdih', "--nangdih", help = "Define new angles and dihedrals required \
                         after amide bond formation.", default = bool(True))
	parser.add_argument('-ff', "--ff", help = "Path to force field files.", default = "amberff99.ff")
	parser.add_argument('-db', "--db", help = "Path to sqlite database.", default = "sqlite_memdb.db")
	parser.add_argument('-log', "--log", help = "Enabling this flag creates a log file.", default = "yes")
	parser.add_argument('-logfile', "--logfile", help = "Log filename.", default = "polymerizeit.log")
	parser.add_argument('-iter', "--iter", help = "minimization-equilibration cycle number.")

	return parser

# Standard boilerplate to call the main() function to begin the program.
if __name__=="__main__":
    try:
		
        args=create_parser().parse_args()
        param_args=vars(args)
        run=GMXsimple(**param_args)
        r_val = run.run()
    except (RuntimeError, TypeError, NameError) as e:
       print(e)
       print("Something went wrong in checking the reactions")	

     
