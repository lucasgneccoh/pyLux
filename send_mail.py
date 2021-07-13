import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess

def parseInputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", help="File containing the body of the message")
    parser.add_argument("--message", help="If body the body of the message is not on a file, then write the message here")
    parser.add_argument("--subject", help="Subject of the message")
    parser.add_argument("--to", help="Destinataries of the message", required = True)
    args = parser.parse_args()
    return args 
    
args = parseInputs()
pwd = os.getcwd()

if "file" in args:
    cmd = "cat"
    # Assuming the entered path is realtive to the current wd
    msg = os.path.join(pwd, args.file)
else:
    if "message" in args:
        cmd = "echo"
        msg = args.message
    else:
        print("Arguments 'file' or 'message' must be given")
        sys.exit(1)

subj = f' -s "{args.subject}"' if "subject" in args else ""

    

command = f'{cmd} {msg} | mail{subj} {args.to}'

tmp_file_name = "tmp_file_mail.sh"

with open(tmp_file_name, "w") as f:
    f.write(command)
    f.close()



subprocess.run(["ssh", "-p", "5022", "lgnecco@lamgate4", "bash", "-s", "<", os.path.join(pwd,tmp_file_name)])

print("Done")