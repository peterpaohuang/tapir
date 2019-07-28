import subprocess
print("Install required packages")
subprocess.check_output(['conda','install','--file','requirements.txt'])
