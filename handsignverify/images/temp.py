import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'handsignverify'])

import handsignverify

handsignverify.handsignverify.match_sign('001_01.PNG','001_02.PNG')