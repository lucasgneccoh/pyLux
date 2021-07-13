import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess

def parseInputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)        
    parser.add_argument("--subject", "-s", help="Subject of the message")
    parser.add_argument("--attach", "-a", help="Files to attach", nargs="*")
    parser.add_argument("--to", help="Destinataries of the message", required = True)
    args = parser.parse_args()
    return args 
    
args = parseInputs()
pwd = os.getcwd()



body = "This mail was sent using Linux and python"
subj = f' -s "{args.subject}"' if "subject" in args else ""
attach = ' '.join([f'-a {os.path.join(pwd,s)}' for s in args.attach]) if "attach" in args else ""

command = f'echo "{body}" | mail{subj}{attach} {args.to}'

print(command)

tmp_file_name = "tmp_file_mail.sh"

with open(tmp_file_name, "w") as f:
    f.write(command)
    f.close()



subprocess.run(["ssh", "-p", "5022", "lgnecco@lamgate4", "bash", "-s", "<", os.path.join(pwd,tmp_file_name)])

print("Done")